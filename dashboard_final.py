import streamlit as st
import pandas as pd
import plotly.express as px
import re

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
        for month in range(1, 13):
            col_name = f"{prefix}{str(month).zfill(2)}"
            if col_name in df.columns:
                value_cols.append(col_name)

    if not value_cols:
        st.error("Lỗi: Không tìm thấy cột nào theo tháng để unpivot.")
        return pd.DataFrame()

    df_melted = pd.melt(df,
                        id_vars=id_cols,
                        value_vars=value_cols,
                        var_name='Metric_Month_Raw',
                        value_name='Value')

    def extract_metric_and_month(metric_month_raw):
        for prefix in sorted(metric_prefixes, key=len, reverse=True):
            if metric_month_raw.startswith(prefix):
                month_str = metric_month_raw[len(prefix):]
                try:
                    month = int(month_str)
                    metric_name = prefix.rstrip('_')
                    return metric_name, month
                except ValueError:
                    pass
        return None, None

    extracted_data = df_melted['Metric_Month_Raw'].apply(extract_metric_and_month)
    df_melted['Metric'] = extracted_data.apply(lambda x: x[0] if x else None)
    df_melted['Month'] = extracted_data.apply(lambda x: x[1] if x else None)
    df_melted.dropna(subset=['Metric', 'Month'], inplace=True)
    df_melted['Month'] = df_melted['Month'].astype(int)

    try:
        df_unpivoted_final = df_melted.pivot_table(
            index=id_cols + ['Month'],
            columns='Metric',
            values='Value'
        ).reset_index()
    except Exception as e:
        st.error(f"Lỗi khi thực hiện pivot_table: {e}")
        return pd.DataFrame()
    
    def sanitize_column_name(col_name):
        if col_name in id_cols + ['Month']:
            return col_name
        name = str(col_name)
        name = name.replace('Đ', 'D').replace('đ', 'd')
        name = name.replace('/', '_')
        name = name.replace(' <', '_lt_')
        name = name.replace('<', '_lt_')
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
        else: 
            if df_unpivoted_final[col].notna().any():
                 try:
                    if col in ['Revenue_DH', 'ASO', 'SKU_ASO', 'SKU_DH', 'Revenue', 'Volume', 'SKU', 'KS_lt_5tr', 'KH_lt_100k']:
                        df_unpivoted_final[col] = pd.to_numeric(df_unpivoted_final[col], errors='coerce')
                 except Exception as e:
                    st.warning(f"Không thể chuyển đổi cột '{col}' thành dạng số hoàn toàn. Lỗi: {e}")

    return df_unpivoted_final

# --- Tải dữ liệu ---
file_path = r'C:\Users\linh.bt\Downloads\cấu trúc dkm.xlsx' 
df_processed = load_and_preprocess_data(file_path, excel_sheet_name='Sheet1') 

# --- Giao diện Streamlit ---
st.title('Dashboard Phân Tích Dữ Liệu Trendline theo NPP')

if df_processed.empty or df_processed.shape[0] == 0 :
    st.warning("Không có dữ liệu để hiển thị. Vui lòng kiểm tra file dữ liệu, đường dẫn và tên sheet.")
else:
    st.sidebar.header('Bộ lọc')
    unique_regions = sorted(list(df_processed['Region'].unique()))
    selected_regions = st.sidebar.multiselect('Chọn Miền:', options=unique_regions, default=list(unique_regions))

    current_available_areas = df_processed['Area'].unique()
    if selected_regions:
        current_available_areas = df_processed[df_processed['Region'].isin(selected_regions)]['Area'].unique()
    available_areas_options = sorted(list(current_available_areas))
    selected_areas = st.sidebar.multiselect('Chọn Vùng:', options=available_areas_options, default=list(available_areas_options))
    
    current_available_npp = df_processed['Tên NPP'].unique()
    if selected_regions and selected_areas:
        current_available_npp = df_processed[
            df_processed['Region'].isin(selected_regions) &
            df_processed['Area'].isin(selected_areas)
        ]['Tên NPP'].unique()
    elif selected_regions:
         current_available_npp = df_processed[df_processed['Region'].isin(selected_regions)]['Tên NPP'].unique()
    available_npp_options = sorted(list(current_available_npp))
    selected_npp = st.sidebar.multiselect('Chọn Tên NPP:', options=available_npp_options, default=list(available_npp_options))
        
    current_available_route_types = df_processed['Loại Route'].unique()
    if selected_regions and selected_areas and selected_npp:
        current_available_route_types = df_processed[
            df_processed['Region'].isin(selected_regions) &
            df_processed['Area'].isin(selected_areas) &
            df_processed['Tên NPP'].isin(selected_npp)
        ]['Loại Route'].unique()
    available_route_type_options = sorted(list(current_available_route_types))
    selected_route_types = st.sidebar.multiselect('Chọn Loại Tuyến:', options=available_route_type_options, default=list(available_route_type_options))

    unique_months = sorted(list(df_processed['Month'].unique()))
    selected_months = st.sidebar.multiselect('Chọn Tháng:',options=unique_months,default=list(unique_months))
    
    df_filtered = df_processed.copy()
    if selected_regions: df_filtered = df_filtered[df_filtered['Region'].isin(selected_regions)]
    if selected_areas: df_filtered = df_filtered[df_filtered['Area'].isin(selected_areas)]
    if selected_npp: df_filtered = df_filtered[df_filtered['Tên NPP'].isin(selected_npp)]
    if selected_route_types: df_filtered = df_filtered[df_filtered['Loại Route'].isin(selected_route_types)]
    if selected_months: df_filtered = df_filtered[df_filtered['Month'].isin(selected_months)]

    if df_filtered.empty:
        st.info("Không có dữ liệu nào phù hợp với các lựa chọn bộ lọc của bạn.")
    else:
        st.success(f"Đã lọc được {len(df_filtered)} dòng dữ liệu.")

        def create_pivot_table_for_display(chart_df, metric_col):
            if chart_df.empty or metric_col not in chart_df.columns:
                return pd.DataFrame(), [] 
            if 'Date' not in chart_df.columns:
                 return chart_df, [] # Trả về bảng gốc nếu không có cột Date
            
            table_data = chart_df.copy()
            table_data['Tháng'] = table_data['Date'].dt.month
            
            try:
                pivot_df = table_data.pivot_table(index='Tên NPP', 
                                                  columns='Tháng', 
                                                  values=metric_col)
                month_cols_renamed = [f"Tháng {int(col)}" for col in pivot_df.columns]
                pivot_df.columns = month_cols_renamed
                return pivot_df.reset_index(), month_cols_renamed 
            except Exception as e:
                # st.warning(f"Không thể tạo bảng pivot cho {metric_col}: {e}") # Tạm ẩn cảnh báo này
                return pd.DataFrame(), []


        # --- Sắp xếp 6 Biểu đồ và Bảng vào các cột ---
        
        # Hàng 1
        col1, col2 = st.columns(2)

        with col1:
            metric_col_chart1 = 'Revenue_DH'
            title_chart1 = '1. Giá trị bình quân đơn hàng/NPP'
            st.subheader(title_chart1)
            if metric_col_chart1 in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[metric_col_chart1]):
                chart_data1 = df_filtered.dropna(subset=[metric_col_chart1]).groupby(['Date', 'Tên NPP'], as_index=False)[metric_col_chart1].mean()
                if not chart_data1.empty:
                    fig1 = px.line(chart_data1, x='Date', y=metric_col_chart1, color='Tên NPP', markers=True,
                                   labels={metric_col_chart1: 'GTBQ ĐH (VND)'}) 
                    fig1.update_yaxes(tickformat=",.0f")
                    fig1.update_layout(title_text=title_chart1, title_x=0.5, height=350) 
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    st.caption("Bảng dữ liệu chi tiết:") # **BỎ EXPANDER**
                    table1_display, month_cols1 = create_pivot_table_for_display(chart_data1, metric_col_chart1)
                    if not table1_display.empty and month_cols1:
                        format_dict1 = {col: "{:,.0f}" for col in month_cols1}
                        st.dataframe(table1_display.style.format(format_dict1), use_container_width=True) # **HIỂN THỊ TRỰC TIẾP**
                else:
                    st.info(f"Không có dữ liệu cho {title_chart1} với lựa chọn hiện tại.")
            else:
                st.warning(f"Cột '{metric_col_chart1}' lỗi hoặc không phải số để vẽ {title_chart1}.")

        with col2:
            metric_col_chart2 = 'ASO'
            title_chart2 = '2. Tổng số ASO/NPP'
            st.subheader(title_chart2)
            if metric_col_chart2 in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[metric_col_chart2]):
                chart_data2 = df_filtered.dropna(subset=[metric_col_chart2]).groupby(['Date', 'Tên NPP'], as_index=False)[metric_col_chart2].sum()
                if not chart_data2.empty:
                    fig2 = px.line(chart_data2, x='Date', y=metric_col_chart2, color='Tên NPP', markers=True,
                                   labels={metric_col_chart2: 'Tổng ASO'})
                    fig2.update_layout(title_text=title_chart2, title_x=0.5, height=350)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("Bảng dữ liệu chi tiết:") # **BỎ EXPANDER**
                    table2_display, _ = create_pivot_table_for_display(chart_data2, metric_col_chart2)
                    if not table2_display.empty:
                        st.dataframe(table2_display, use_container_width=True) # **HIỂN THỊ TRỰC TIẾP**
                else:
                    st.info(f"Không có dữ liệu cho {title_chart2} với lựa chọn hiện tại.")
            else:
                st.warning(f"Cột '{metric_col_chart2}' lỗi hoặc không phải số để vẽ {title_chart2}.")

        # Hàng 2
        col3, col4 = st.columns(2)

        with col3:
            metric_col_chart3 = 'SKU_ASO'
            title_chart3 = '3. Số SKU/ASO/NPP'
            st.subheader(title_chart3)
            if metric_col_chart3 in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[metric_col_chart3]):
                chart_data3 = df_filtered.dropna(subset=[metric_col_chart3]).groupby(['Date', 'Tên NPP'], as_index=False)[metric_col_chart3].mean()
                if not chart_data3.empty:
                    fig3 = px.line(chart_data3, x='Date', y=metric_col_chart3, color='Tên NPP', markers=True,
                                   labels={metric_col_chart3: 'SKU/ASO (TB)'})
                    fig3.update_yaxes(tickformat=".1f")
                    fig3.update_layout(title_text=title_chart3, title_x=0.5, height=350)
                    st.plotly_chart(fig3, use_container_width=True)
                    st.caption("Bảng dữ liệu chi tiết:") # **BỎ EXPANDER**
                    table3_display, month_cols3 = create_pivot_table_for_display(chart_data3, metric_col_chart3)
                    if not table3_display.empty and month_cols3:
                        format_dict3 = {col: "{:.1f}" for col in month_cols3}
                        st.dataframe(table3_display.style.format(format_dict3), use_container_width=True) # **HIỂN THỊ TRỰC TIẾP**
                else:
                    st.info(f"Không có dữ liệu cho {title_chart3} với lựa chọn hiện tại.")
            else:
                st.warning(f"Cột '{metric_col_chart3}' lỗi hoặc không phải số để vẽ {title_chart3}.")

        with col4:
            metric_col_chart4 = 'SKU_DH'
            title_chart4 = '4. Số SKU/ĐH/NPP'
            st.subheader(title_chart4)
            if metric_col_chart4 in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[metric_col_chart4]):
                chart_data4 = df_filtered.dropna(subset=[metric_col_chart4]).groupby(['Date', 'Tên NPP'], as_index=False)[metric_col_chart4].mean()
                if not chart_data4.empty:
                    fig4 = px.line(chart_data4, x='Date', y=metric_col_chart4, color='Tên NPP', markers=True,
                                   labels={metric_col_chart4: 'SKU/ĐH (TB)'})
                    fig4.update_yaxes(tickformat=".1f")
                    fig4.update_layout(title_text=title_chart4, title_x=0.5, height=350)
                    st.plotly_chart(fig4, use_container_width=True)
                    st.caption("Bảng dữ liệu chi tiết:") # **BỎ EXPANDER**
                    table4_display, month_cols4 = create_pivot_table_for_display(chart_data4, metric_col_chart4)
                    if not table4_display.empty and month_cols4:
                        format_dict4 = {col: "{:.1f}" for col in month_cols4}
                        st.dataframe(table4_display.style.format(format_dict4), use_container_width=True) # **HIỂN THỊ TRỰC TIẾP**
                else:
                    st.info(f"Không có dữ liệu cho {title_chart4} với lựa chọn hiện tại.")
            else:
                st.warning(f"Cột '{metric_col_chart4}' lỗi hoặc không phải số để vẽ {title_chart4}.")

        # Hàng 3
        col5, col6 = st.columns(2)
        
        with col5:
            metric_col_chart5 = 'KS_lt_5tr'
            title_chart5 = '5. SL KH (KS <5tr)/NPP'
            st.subheader(title_chart5)
            if metric_col_chart5 in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[metric_col_chart5]):
                chart_data5 = df_filtered.dropna(subset=[metric_col_chart5]).groupby(['Date', 'Tên NPP'], as_index=False)[metric_col_chart5].sum()
                if not chart_data5.empty:
                    fig5 = px.line(chart_data5, x='Date', y=metric_col_chart5, color='Tên NPP', markers=True,
                                   labels={metric_col_chart5: 'SL KH (KS <5tr)'})
                    fig5.update_layout(title_text=title_chart5, title_x=0.5, height=350)
                    st.plotly_chart(fig5, use_container_width=True)
                    st.caption("Bảng dữ liệu chi tiết:") # **BỎ EXPANDER**
                    table5_display, _ = create_pivot_table_for_display(chart_data5, metric_col_chart5)
                    if not table5_display.empty:
                        st.dataframe(table5_display, use_container_width=True) # **HIỂN THỊ TRỰC TIẾP**
                else:
                    st.info(f"Không có dữ liệu cho {title_chart5} với lựa chọn hiện tại.")
            else:
                st.warning(f"Cột '{metric_col_chart5}' lỗi hoặc không phải số để vẽ {title_chart5}.")
        
        with col6:
            metric_col_chart6 = 'KH_lt_100k'
            title_chart6 = '6. SL KH (DS <100k)/NPP'
            st.subheader(title_chart6)
            if metric_col_chart6 in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[metric_col_chart6]):
                chart_data6 = df_filtered.dropna(subset=[metric_col_chart6]).groupby(['Date', 'Tên NPP'], as_index=False)[metric_col_chart6].sum()
                if not chart_data6.empty:
                    fig6 = px.line(chart_data6, x='Date', y=metric_col_chart6, color='Tên NPP', markers=True,
                                   labels={metric_col_chart6: 'SL KH (DS <100k)'})
                    fig6.update_layout(title_text=title_chart6, title_x=0.5, height=350)
                    st.plotly_chart(fig6, use_container_width=True)
                    st.caption("Bảng dữ liệu chi tiết:") # **BỎ EXPANDER**
                    table6_display, _ = create_pivot_table_for_display(chart_data6, metric_col_chart6)
                    if not table6_display.empty:
                        st.dataframe(table6_display, use_container_width=True) # **HIỂN THỊ TRỰC TIẾP**
                else:
                    st.info(f"Không có dữ liệu cho {title_chart6} với lựa chọn hiện tại.")
            else:
                st.warning(f"Cột '{metric_col_chart6}' lỗi hoặc không phải dạng số để vẽ {title_chart6}.")
