import streamlit as st
import pandas as pd
import plotly.express as px
import re
from dateutil.relativedelta import relativedelta # Để tính tháng trước

# --- Configuration ---
st.set_page_config(layout="wide")
YEAR_DATA = 2025

@st.cache_data
def load_and_preprocess_data(file_path, excel_sheet_name='Sheet1'):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, sheet_name=excel_sheet_name, engine='openpyxl')
        else:
            st.error(f"Lỗi: Định dạng file '{file_path}' không được hỗ trợ.")
            return pd.DataFrame()
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file '{file_path}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return pd.DataFrame()

    id_cols = ['Region', 'Area', 'Tên NPP', 'Mã Route', 'Loại Route']
    missing_id_cols = [col for col in id_cols if col not in df.columns]
    if missing_id_cols:
        st.error(f"Lỗi: Các cột định danh sau không có trong file: {', '.join(missing_id_cols)}.")
        return pd.DataFrame()

    metric_prefixes = [
        'Revenue/ĐH_', 'ASO_', 'SKU/ASO_', 'SKU/ĐH_',
        'KS <5tr_', 'KH <100k_', 'Revenue_', 'Volume_', 'SKU_'
    ]
    value_cols = []
    for prefix in metric_prefixes:
        for month_num in range(1, 13):
            col_name = f"{prefix}{str(month_num).zfill(2)}"
            if col_name in df.columns:
                value_cols.append(col_name)

    if not value_cols:
        st.error("Lỗi: Không tìm thấy cột nào theo tháng để unpivot.")
        return pd.DataFrame()

    df_melted = pd.melt(df, id_vars=id_cols, value_vars=value_cols,
                        var_name='Metric_Month_Raw', value_name='Value')

    def extract_metric_and_month(metric_month_raw):
        for prefix in sorted(metric_prefixes, key=len, reverse=True):
            if metric_month_raw.startswith(prefix):
                month_str = metric_month_raw[len(prefix):]
                try:
                    month = int(month_str)
                    metric_name = prefix.rstrip('_')
                    return metric_name, month
                except ValueError: pass
        return None, None

    extracted_data = df_melted['Metric_Month_Raw'].apply(extract_metric_and_month)
    df_melted['Metric'] = extracted_data.apply(lambda x: x[0] if x else None)
    df_melted['Month'] = extracted_data.apply(lambda x: x[1] if x else None)
    df_melted.dropna(subset=['Metric', 'Month'], inplace=True)
    df_melted['Month'] = df_melted['Month'].astype(int) # Sửa lỗi chính tả ở đây

    try:
        df_unpivoted_final = df_melted.pivot_table(index=id_cols + ['Month'],
                                                columns='Metric', values='Value').reset_index()
    except Exception as e:
        st.error(f"Lỗi khi thực hiện pivot_table: {e}")
        return pd.DataFrame()
    
    def sanitize_column_name(col_name):
        if col_name in id_cols + ['Month']: return col_name
        name = str(col_name).replace('Đ', 'D').replace('đ', 'd').replace('/', '_')
        name = name.replace(' <', '_lt_').replace('<', '_lt_')
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        return name

    df_unpivoted_final.columns = [sanitize_column_name(col) for col in df_unpivoted_final.columns]
    df_unpivoted_final['Year'] = YEAR_DATA
    try:
        df_unpivoted_final['Date'] = pd.to_datetime(
            df_unpivoted_final['Year'].astype(str) + '-' + df_unpivoted_final['Month'].astype(str) + '-01',
            format='%Y-%m-%d'
        )
    except Exception as e:
        st.error(f"Lỗi khi tạo cột 'Date': {e}")
        return df_unpivoted_final
    
    expected_metric_cols = ['Revenue_DH', 'ASO', 'SKU_ASO', 'SKU_DH', 'KS_lt_5tr', 'KH_lt_100k', 'Revenue', 'Volume', 'SKU']
    for col in expected_metric_cols:
        if col not in df_unpivoted_final.columns:
            df_unpivoted_final[col] = pd.NA
        elif df_unpivoted_final[col].notna().any():
            try:
                df_unpivoted_final[col] = pd.to_numeric(df_unpivoted_final[col], errors='coerce')
            except Exception as e:
                st.warning(f"Không thể chuyển đổi cột '{col}' thành dạng số hoàn toàn. Lỗi: {e}")
    return df_unpivoted_final

# --- Tải dữ liệu ---
file_path = 'cấu trúc dkm.xlsx' 
df_processed = load_and_preprocess_data(file_path, excel_sheet_name='Sheet1') 

# --- Giao diện Streamlit ---
st.title('Dashboard Phân Tích Dữ Liệu')

if df_processed.empty or df_processed.shape[0] == 0 :
    st.warning("Không có dữ liệu để hiển thị. Vui lòng kiểm tra file dữ liệu, đường dẫn và tên sheet.")
else:
    st.sidebar.header('Bộ lọc Báo cáo')

    reporting_object_options = ["Nhà Phân Phối", "Miền", "Vùng"]
    reporting_object = st.sidebar.selectbox(
        "Chọn đối tượng để chạy báo cáo:", 
        reporting_object_options, 
        key="reporting_object_selector"
    )
    st.sidebar.markdown("---")

    unique_regions_all = sorted(list(df_processed['Region'].unique()))
    
    selected_regions_filter_ui = [] 
    selected_areas_filter_ui = []   
    selected_npp_filter_ui = []     
    selected_route_types_filter_ui = []
    primary_selected_objects_for_analysis = []

    unique_months_for_charts = sorted(list(df_processed['Month'].unique()))
    selected_months_for_charts = st.sidebar.multiselect(
        'Phạm vi tháng cho biểu đồ:',
        options=unique_months_for_charts,
        default=list(unique_months_for_charts),
        key="months_for_charts_multiselect"
    )

    month_year_strings = sorted(df_processed['Date'].dt.strftime('%m/%Y').unique(), 
                                key=lambda x: (int(x.split('/')[1]), int(x.split('/')[0])))
    default_mom_month_idx = len(month_year_strings) - 1 if month_year_strings else 0
    selected_mom_month_str = st.sidebar.selectbox(
        "Chọn tháng để so sánh (MoM):", 
        options=month_year_strings, 
        index=default_mom_month_idx,
        key="mom_month_selectbox"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Bộ lọc chi tiết:")

    if reporting_object == "Nhà Phân Phối":
        selected_regions_filter_ui = st.sidebar.multiselect('Chọn Miền:', options=unique_regions_all, default=list(unique_regions_all), key="region_filter_npp_mode")
        
        areas_to_offer_npp_mode = df_processed
        if selected_regions_filter_ui: areas_to_offer_npp_mode = areas_to_offer_npp_mode[areas_to_offer_npp_mode['Region'].isin(selected_regions_filter_ui)]
        unique_areas_npp_mode = sorted(list(areas_to_offer_npp_mode['Area'].unique()))
        selected_areas_filter_ui = st.sidebar.multiselect('Chọn Vùng:', options=unique_areas_npp_mode, default=list(unique_areas_npp_mode), key="area_filter_npp_mode")

        npp_to_offer_npp_mode = areas_to_offer_npp_mode 
        if selected_areas_filter_ui: npp_to_offer_npp_mode = npp_to_offer_npp_mode[npp_to_offer_npp_mode['Area'].isin(selected_areas_filter_ui)]
        unique_npp_npp_mode = sorted(list(npp_to_offer_npp_mode['Tên NPP'].unique()))
        selected_npp_filter_ui = st.sidebar.multiselect('Chọn Tên NPP:', options=unique_npp_npp_mode, default=list(unique_npp_npp_mode), key="npp_filter_npp_mode")
        primary_selected_objects_for_analysis = selected_npp_filter_ui
        
        route_to_offer_npp_mode = npp_to_offer_npp_mode 
        if selected_npp_filter_ui: route_to_offer_npp_mode = route_to_offer_npp_mode[route_to_offer_npp_mode['Tên NPP'].isin(selected_npp_filter_ui)]
        unique_routes_npp_mode = sorted(list(route_to_offer_npp_mode['Loại Route'].unique()))
        selected_route_types_filter_ui = st.sidebar.multiselect('Chọn Loại Tuyến:', options=unique_routes_npp_mode, default=list(unique_routes_npp_mode), key="route_filter_npp_mode")

    elif reporting_object == "Miền":
        primary_selected_objects_for_analysis = st.sidebar.multiselect('Chọn Miền để phân tích/hiển thị:', options=unique_regions_all, default=list(unique_regions_all), key="region_analysis_mode")
        selected_regions_filter_ui = primary_selected_objects_for_analysis 

    elif reporting_object == "Vùng":
        selected_regions_filter_ui = st.sidebar.multiselect('Lọc Vùng theo Miền (tùy chọn):', options=unique_regions_all, default=[], key="region_filter_for_area_selection")
        
        areas_to_offer_area_mode = df_processed
        if selected_regions_filter_ui: areas_to_offer_area_mode = areas_to_offer_area_mode[areas_to_offer_area_mode['Region'].isin(selected_regions_filter_ui)]
        unique_areas_area_mode = sorted(list(areas_to_offer_area_mode['Area'].unique()))
        primary_selected_objects_for_analysis = st.sidebar.multiselect('Chọn Vùng để phân tích/hiển thị:', options=unique_areas_area_mode, default=list(unique_areas_area_mode), key="area_analysis_mode")
        selected_areas_filter_ui = primary_selected_objects_for_analysis

    group_by_col_data = None
    color_col_data = None
    group_by_col_name_for_display = "" 

    if reporting_object == "Nhà Phân Phối":
        group_by_col_data = 'Tên NPP'
        color_col_data = 'Tên NPP'
        group_by_col_name_for_display = "Nhà Phân Phối"
    elif reporting_object == "Miền":
        group_by_col_data = 'Region'
        color_col_data = 'Region'
        group_by_col_name_for_display = "Miền"
    elif reporting_object == "Vùng":
        group_by_col_data = 'Area'
        color_col_data = 'Area'
        group_by_col_name_for_display = "Vùng"

    df_to_filter = df_processed.copy()
    if reporting_object == "Nhà Phân Phối":
        if selected_regions_filter_ui: df_to_filter = df_to_filter[df_to_filter['Region'].isin(selected_regions_filter_ui)]
        if selected_areas_filter_ui: df_to_filter = df_to_filter[df_to_filter['Area'].isin(selected_areas_filter_ui)]
        if selected_npp_filter_ui: df_to_filter = df_to_filter[df_to_filter['Tên NPP'].isin(selected_npp_filter_ui)]
        if selected_route_types_filter_ui: df_to_filter = df_to_filter[df_to_filter['Loại Route'].isin(selected_route_types_filter_ui)]
    elif reporting_object == "Miền":
        if primary_selected_objects_for_analysis: df_to_filter = df_to_filter[df_to_filter['Region'].isin(primary_selected_objects_for_analysis)]
    elif reporting_object == "Vùng":
        if selected_regions_filter_ui : df_to_filter = df_to_filter[df_to_filter['Region'].isin(selected_regions_filter_ui)] 
        if primary_selected_objects_for_analysis: df_to_filter = df_to_filter[df_to_filter['Area'].isin(primary_selected_objects_for_analysis)]
        
    df_for_charts = df_to_filter.copy()
    if selected_months_for_charts:
        df_for_charts = df_for_charts[df_for_charts['Month'].isin(selected_months_for_charts)]
    else: 
        df_for_charts = pd.DataFrame(columns=df_to_filter.columns)

    if df_for_charts.empty and df_to_filter.empty : 
        st.info("Không có dữ liệu nào phù hợp với các lựa chọn bộ lọc của bạn.")
    else:
        if not df_for_charts.empty :
            pass
        elif not df_to_filter.empty:
            st.info("Vui lòng chọn phạm vi tháng cho biểu đồ để hiển thị.")

        def create_pivot_table_for_display(chart_df, metric_col, index_col_for_pivot):
            if chart_df.empty or metric_col not in chart_df.columns or index_col_for_pivot not in chart_df.columns:
                return pd.DataFrame(), [] 
            if 'Date' not in chart_df.columns:
                 return chart_df, []
            
            table_data = chart_df.copy()
            table_data['Tháng'] = table_data['Date'].dt.month
            
            try:
                pivot_df = table_data.pivot_table(index=index_col_for_pivot, columns='Tháng', values=metric_col)
                month_cols_renamed = [f"Tháng {int(col)}" for col in pivot_df.columns]
                pivot_df.columns = month_cols_renamed
                return pivot_df.reset_index(), month_cols_renamed 
            except Exception as e:
                return pd.DataFrame(), []

        st.markdown("---")
        st.header(f"Phân tích Trendline theo {group_by_col_name_for_display}")

        col_defs = [
            {'metric': 'Revenue_DH', 'title_template': '1. Giá trị bình quân đơn hàng/{}', 'agg': 'mean', 'y_label_template': 'GTBQ ĐH ({})', 'tick_format': ',.0f'},
            {'metric': 'ASO', 'title_template': '2. Tổng số ASO/{}', 'agg': 'sum', 'y_label_template': 'Tổng ASO ({})', 'tick_format': None, 'table_format': '{:.0f}'},
            {'metric': 'SKU_ASO', 'title_template': '3. Số SKU/ASO/{}', 'agg': 'mean', 'y_label_template': 'SKU/ASO (TB {})', 'tick_format': '.1f'},
            {'metric': 'SKU_DH', 'title_template': '4. Số SKU/ĐH/{}', 'agg': 'mean', 'y_label_template': 'SKU/ĐH (TB {})', 'tick_format': '.1f'},
            {'metric': 'KS_lt_5tr', 'title_template': '5. SL KH (KS <5tr)/{}', 'agg': 'sum', 'y_label_template': 'SL KH (KS <5tr) ({})', 'tick_format': None, 'table_format': '{:.0f}'}, # ĐÃ THÊM table_format
            {'metric': 'KH_lt_100k', 'title_template': '6. SL KH (DS <100k)/{}', 'agg': 'sum', 'y_label_template': 'SL KH (DS <100k) ({})', 'tick_format': None, 'table_format': '{:.0f}'}  # ĐÃ THÊM table_format
        ]

        if not df_for_charts.empty and group_by_col_data and color_col_data:
            cols_per_row = 2
            for i in range(0, len(col_defs), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(col_defs):
                        chart_config = col_defs[i+j]
                        with cols[j]:
                            metric_col = chart_config['metric']
                            title_chart = chart_config['title_template'].format(group_by_col_name_for_display)
                            y_label = chart_config['y_label_template'].format(group_by_col_name_for_display.lower())
                            agg_method = chart_config['agg']
                            tick_format_chart = chart_config['tick_format']
                            table_format_metric = chart_config.get('table_format')

                            st.subheader(title_chart)
                            if metric_col in df_for_charts.columns and pd.api.types.is_numeric_dtype(df_for_charts[metric_col]) and group_by_col_data in df_for_charts.columns:
                                if agg_method == 'mean':
                                    chart_data = df_for_charts.dropna(subset=[metric_col]).groupby(['Date', group_by_col_data], as_index=False)[metric_col].mean()
                                elif agg_method == 'sum':
                                    chart_data = df_for_charts.dropna(subset=[metric_col]).groupby(['Date', group_by_col_data], as_index=False)[metric_col].sum()
                                else: 
                                    chart_data = df_for_charts.dropna(subset=[metric_col]).groupby(['Date', group_by_col_data], as_index=False)[metric_col].mean()
                                
                                if not chart_data.empty:
                                    unique_color_values = chart_data[color_col_data].unique()
                                    color_discrete_map_for_chart = {
                                        val: px.colors.qualitative.Plotly[k % len(px.colors.qualitative.Plotly)]
                                        for k, val in enumerate(unique_color_values)
                                    }

                                    fig = px.line(chart_data, x='Date', y=metric_col, color=color_col_data, 
                                                  markers=True, labels={metric_col: y_label, color_col_data: group_by_col_name_for_display},
                                                  color_discrete_map=color_discrete_map_for_chart)
                                    
                                    if tick_format_chart:
                                        fig.update_yaxes(tickformat=tick_format_chart)
                                    fig.update_layout(title_text="", title_x=0.5, height=350)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.caption("Bảng dữ liệu chi tiết:")
                                    table_display, month_cols_display = create_pivot_table_for_display(chart_data, metric_col, index_col_for_pivot=group_by_col_data)
                                    
                                    if not table_display.empty:
                                        npp_color_map_for_table = color_discrete_map_for_chart.copy()
                                        
                                        # **LOGIC TẠO table_format_dict ĐÃ ĐƯỢC SỬA**
                                        table_format_dict = {}
                                        month_col_formatter = None # Định dạng cho các cột tháng

                                        if table_format_metric: # Ưu tiên 1: table_format từ col_defs
                                            month_col_formatter = table_format_metric
                                        elif tick_format_chart == ",.0f": # Ưu tiên 2: dựa trên tick_format của chart
                                            month_col_formatter = "{:,.0f}"
                                        elif tick_format_chart == ".1f":
                                            month_col_formatter = "{:.1f}"
                                        # Thêm các elif khác ở đây nếu có nhiều loại tick_format cần chuyển đổi cụ thể
                                        
                                        # Nếu sau các bước trên month_col_formatter vẫn là None (ví dụ: tick_format là None và table_format không có)
                                        # thì áp dụng một định dạng mặc định chung cho các cột tháng.
                                        if month_col_formatter is None and month_cols_display:
                                            month_col_formatter = "{:,.0f}" # Mặc định: số nguyên, có dấu phẩy

                                        # Áp dụng định dạng cho các cột tháng
                                        if month_col_formatter and month_cols_display:
                                            for col_m_display in month_cols_display:
                                                table_format_dict[col_m_display] = month_col_formatter
                                        
                                        # Định dạng cột đối tượng (NPP/Miền/Vùng) với màu sắc
                                        table_format_dict[group_by_col_data] = lambda val_in_table: f'<span style="display:inline-block; border-radius:3px; margin-right:7px; width:12px; height:12px; background-color:{npp_color_map_for_table.get(val_in_table, "grey")}; vertical-align:middle;"></span>{val_in_table}'
                                        
                                        styled_table = table_display.style.format(table_format_dict).hide(axis="index")
                                        html_table = styled_table.set_table_attributes('style="width:100%; overflow-x:auto; display:block;"').to_html()
                                        st.markdown(html_table, unsafe_allow_html=True)
                                else:
                                    st.info(f"Không có dữ liệu cho {title_chart} với lựa chọn hiện tại.")
                            else:
                                st.warning(f"Cột '{metric_col}' hoặc '{group_by_col_data}' không hợp lệ để vẽ {title_chart}.")
        
        # --- PHÂN TÍCH MONTH-OVER-MONTH (MoM) ---
        st.markdown("---")
        st.header(f"Phân tích Tăng trưởng Tháng ({selected_mom_month_str}) so với Tháng trước")

        if selected_mom_month_str and group_by_col_data:
            current_month_period = pd.to_datetime(selected_mom_month_str, format='%m/%Y').to_period('M')
            previous_month_period = current_month_period - 1
            
            df_mom_analysis_base = df_to_filter[
                df_to_filter['Date'].dt.to_period('M').isin([current_month_period, previous_month_period])
            ]

            if df_mom_analysis_base.empty or len(df_mom_analysis_base['Date'].dt.to_period('M').unique()) < 2:
                st.info(f"Không đủ dữ liệu cho tháng {selected_mom_month_str} và/hoặc tháng trước đó ({previous_month_period.strftime('%m/%Y')}) để phân tích MoM với các bộ lọc hiện tại.")
            else:
                items_to_analyze_mom = []
                # Xác định danh sách các đối tượng (item_name) cần phân tích MoM
                # dựa trên reporting_object và các lựa chọn filter đang active cho đối tượng đó
                if reporting_object == "Nhà Phân Phối":
                    # Nếu có NPP được chọn trong selected_npp_filter_ui, chỉ phân tích các NPP đó
                    # Nếu không, phân tích tất cả NPP trong df_to_filter (đã được lọc theo Miền/Vùng nếu có)
                    items_to_analyze_mom = selected_npp_filter_ui if selected_npp_filter_ui else sorted(list(df_to_filter[group_by_col_data].unique()))
                elif reporting_object == "Miền":
                    items_to_analyze_mom = primary_selected_objects_for_analysis if primary_selected_objects_for_analysis else sorted(list(df_to_filter[group_by_col_data].unique()))
                elif reporting_object == "Vùng":
                    items_to_analyze_mom = primary_selected_objects_for_analysis if primary_selected_objects_for_analysis else sorted(list(df_to_filter[group_by_col_data].unique()))
                
                if not items_to_analyze_mom:
                    st.warning(f"Vui lòng chọn ít nhất một '{group_by_col_name_for_display}' từ bộ lọc chi tiết để xem phân tích MoM, hoặc bỏ trống bộ lọc chi tiết để xem tất cả các đối tượng thuộc phạm vi đã chọn.")
                else:
                    for item_name in items_to_analyze_mom:
                        if item_name not in df_mom_analysis_base[group_by_col_data].unique():
                            continue 

                        with st.expander(f"Phân tích MoM cho {group_by_col_name_for_display}: {item_name}"):
                            df_item_mom = df_mom_analysis_base[df_mom_analysis_base[group_by_col_data] == item_name]
                            
                            if df_item_mom.empty:
                                st.write("Không có dữ liệu cho đối tượng này trong kỳ phân tích MoM.")
                                continue

                            for metric_config in col_defs:
                                metric = metric_config['metric']
                                agg_method = metric_config['agg']
                                base_metric_label = metric_config['title_template'].split('. ')[1].format('')[:-1] 
                                
                                if metric not in df_item_mom.columns:
                                    st.caption(f"Không có dữ liệu cho chỉ số '{base_metric_label}'.")
                                    continue

                                if agg_method == 'sum':
                                    monthly_aggregated_metric = df_item_mom.groupby(df_item_mom['Date'].dt.to_period('M'))[metric].sum().reindex([previous_month_period, current_month_period])
                                else: 
                                    monthly_aggregated_metric = df_item_mom.groupby(df_item_mom['Date'].dt.to_period('M'))[metric].mean().reindex([previous_month_period, current_month_period])
                                
                                current_val = monthly_aggregated_metric.get(current_month_period)
                                prev_val = monthly_aggregated_metric.get(previous_month_period)
                                current_val_calc = 0 if pd.isna(current_val) else current_val
                                prev_val_calc = 0 if pd.isna(prev_val) else prev_val
                                
                                delta_text = "N/A"
                                change_desc_suffix = "do không có dữ liệu tháng trước hoặc cả hai tháng bằng 0."

                                if pd.notna(prev_val) and prev_val_calc != 0:
                                    percentage_change = ((current_val_calc - prev_val_calc) / prev_val_calc) * 100
                                    delta_text = f"{percentage_change:.1f}%"
                                    if percentage_change > 0: change_desc_suffix = f"tăng {abs(percentage_change):.1f}%."
                                    elif percentage_change < 0: change_desc_suffix = f"giảm {abs(percentage_change):.1f}%."
                                    else: change_desc_suffix = "không thay đổi."
                                elif pd.notna(prev_val) and prev_val_calc == 0 and current_val_calc != 0:
                                    delta_text = "Tăng mạnh"
                                    change_desc_suffix = "tăng mạnh (tháng trước là 0)."
                                elif pd.notna(prev_val) and prev_val_calc == 0 and current_val_calc == 0:
                                    delta_text = "0.0%" 
                                    change_desc_suffix = "không thay đổi (cả hai tháng là 0)."
                                elif pd.isna(prev_val) and pd.notna(current_val) and current_val_calc != 0 : 
                                    delta_text = "Mới"
                                    change_desc_suffix = "mới xuất hiện (không có dữ liệu tháng trước)."
                                elif pd.isna(prev_val) and pd.notna(current_val) and current_val_calc == 0 :
                                    delta_text = "N/A" 
                                    change_desc_suffix = " (giá trị hiện tại là 0, không có dữ liệu tháng trước)."

                                formatted_current_val = f"{current_val_calc}" 
                                config_table_format_mom = metric_config.get('table_format')
                                config_tick_format_mom = metric_config.get('tick_format')

                                try: # **ĐẢM BẢO ĐỊNH DẠNG ĐÚNG CHO ST.METRIC VALUE**
                                    if config_table_format_mom: 
                                        formatted_current_val = config_table_format_mom.format(current_val_calc)
                                    elif config_tick_format_mom: 
                                        formatted_current_val = f"{current_val_calc:{config_tick_format_mom}}" # Sử dụng f-string đúng
                                    else: 
                                        if isinstance(current_val_calc, float):
                                            if current_val_calc == int(current_val_calc): formatted_current_val = f"{int(current_val_calc):,}"
                                            else: formatted_current_val = f"{current_val_calc:,.2f}" 
                                        else: 
                                            formatted_current_val = f"{current_val_calc:,}"
                                except (ValueError, TypeError): 
                                    formatted_current_val = str(current_val_calc) 

                                st.metric(label=f"{base_metric_label}", 
                                          value=formatted_current_val, 
                                          delta=delta_text if delta_text not in ["0.0%", "N/A"] else None)
                                st.caption(f"Chỉ số '{base_metric_label}' của {group_by_col_name_for_display} '{item_name}' {change_desc_suffix}")
                                st.markdown("""<hr style="margin-top:0.5rem; margin-bottom:0.5rem;" />""", unsafe_allow_html=True)
        else:
            st.info("Vui lòng chọn một tháng để thực hiện phân tích so sánh.")
