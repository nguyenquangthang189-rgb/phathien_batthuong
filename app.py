import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import io

# ==========================================
# CẤU HÌNH TRANG ĐẦU TIÊN (MANDATORY)
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# ==========================================
# CÁC HÀM CACHE & BỔ TRỢ DÙNG CHUNG
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để đảm bảo khả năng hash của Streamlit cache"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Định nghĩa danh sách các biến đặc trưng dựa theo Notebook và Dataset
FEATURES = [f'X_{i}' for i in range(1, 15)]
TARGET = 'default'

# ==========================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn file dataset1.csv hoặc cấu trúc tương tự để huấn luyện mô hình."
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: Random Forest Classifier (Model3 từ Notebook)")
    
    # Các siêu tham số lấy từ cấu hình chuẩn RandomForest
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây quyết định trong rừng ensemble."
    )
    
    criterion = st.selectbox(
        "Tiêu chí đánh giá (criterion)",
        options=["gini", "entropy", "log_loss"],
        index=0,
        help="Hàm đo lường chất lượng phân tách các nhánh cây."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=1, max_value=30, value=10,
        help="Độ sâu tối đa giới hạn của các cây quyết định nhằm tránh overfitting."
    )
    
    with st.expander("Tham số nâng cao"):
        min_samples_split = st.slider(
            "Mẫu tối thiểu để phân tách", 
            min_value=2, max_value=10, value=2
        )
        random_state = st.number_input(
            "Cài đặt Random State", 
            value=42, step=1,
            help="Đảm bảo tính tái lập kết quả phân tách và huấn luyện."
        )
        test_size = st.slider(
            "Tỷ lệ dữ liệu kiểm định (Test Size)", 
            min_value=0.1, max_value=0.5, value=0.3, step=0.05,
            help="Tỷ lệ chia dữ liệu làm tập Test để đánh giá khách quan mô hình."
        )

    st.divider()
    # Nút bấm hành động kích hoạt huấn luyện duy nhất
    trigger_train = st.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để chạy quy trình tiền xử lý và huấn luyện mô hình dựa trên tham số đã chọn."
    )

# ==========================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🛡️ Ứng dụng Phát hiện Giao dịch Gian lận & Rủi ro Tín dụng")
st.caption("Ứng dụng tự động hóa quy trình phân tích dữ liệu, huấn luyện mô hình Random Forest từ dữ liệu lịch sử và dự báo rủi ro giao dịch trực tuyến.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp dữ liệu ở thanh điều hướng bên trái (Sidebar) để bắt đầu sử dụng ứng dụng.")
    st.stop()
else:
    # Đọc dữ liệu thô qua hàm cache
    file_bytes = uploaded_file.getvalue()
    df_raw = load_data(file_bytes, uploaded_file.name)
    
    if df_raw is None:
        st.error("Không thể đọc tệp dữ liệu. Vui lòng kiểm tra định dạng.")
        st.stop()
        
    st.caption(f"📁 Đang dùng tệp dữ liệu: `{uploaded_file.name}`")

st.divider()

# ==========================================
# KHỐI HUẤN LUYỆN (Chạy khi bấm nút, lưu vào session_state)
# ==========================================
if trigger_train:
    with st.spinner("🔄 Đang xử lý dữ liệu và huấn luyện mô hình..."):
        # Kiểm tra xem dữ liệu có đủ các cột cần thiết không
        missing_cols = [col for col in FEATURES + [TARGET] if col not in df_raw.columns]
        if missing_cols:
            st.error(f"Dữ liệu thiếu các cột bắt buộc: {missing_cols}")
        else:
            # Tách X, y
            X = df_raw[FEATURES]
            y = df_raw[TARGET]
            
            # Chia tập huấn luyện/kiểm định
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Khởi tạo và khớp bộ tiền xử lý Scaler giống như notebook
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # SỬA LỖI WARNING: Khôi phục lại định dạng DataFrame kèm tên cột sau khi scale dữ liệu kiểm định
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURES)
            
            # Khởi tạo mô hình theo tham số người dùng chọn ở sidebar
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Dự đoán sử dụng DataFrame có đầy đủ tên cột đã sửa lỗi
            y_pred = model.predict(X_test_scaled_df)
            y_prob = model.predict_proba(X_test_scaled_df)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Lưu trữ 3 đối tượng quan trọng vào st.session_state theo quy tắc
            st.session_state['trained_model'] = model
            st.session_state['data_scaler'] = scaler
            st.session_state['evaluation_metrics'] = {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'features_importance': model.feature_importances_
            }
            st.success("🎉 Huấn luyện mô hình thành công! Kết quả đã sẵn sàng ở các Tab bên dưới.")

# ==========================================
# KHỞI TẠO HỆ THỐNG TAB CHỨA NỘI DUNG CHÍNH
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện & Kiểm định", 
    "🔮 Dự báo & Sử dụng mô hình"
])

# ------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------
with tab1:
    st.subheader("Phân tích cấu trúc dữ liệu thô")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số dòng (Samples)", f"{df_raw.shape[0]:,}")
    with col2:
        st.metric("Số cột (Features)", f"{df_raw.shape[1]:,}")
    with col3:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
        
    st.markdown("### 📋 5 hàng dữ liệu đầu tiên (Head)")
    st.dataframe(df_raw.head(), use_container_width=True)
    
    st.markdown("### 📉 Thống kê mô tả các biến đặc trưng mô hình")
    # Chỉ hiển thị thống kê đối với các biến đưa vào mô hình (X và y) theo mô tả quy tắc
    available_model_cols = [col for col in FEATURES + [TARGET] if col in df_raw.columns]
    if available_model_cols:
        st.dataframe(df_raw[available_model_cols].describe().T, use_container_width=True)
    else:
        st.warning("Không tìm thấy các biến đặc trưng X_1 đến X_14 trong file dữ liệu.")

# ------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------
with tab2:
    st.subheader("Biểu đồ phân tích phân phối thuộc tính")
    
    # Ưu tiên kiểm tra mục tiêu 'default'
    if TARGET in df_raw.columns:
        # Lưới 2x2 cho cân đối trực quan
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        # 1. Biểu đồ phân phối biến mục tiêu (y)
        with row1_col1:
            target_counts = df_raw[TARGET].value_counts().reset_index()
            target_counts.columns = ['Trạng thái (default)', 'Số lượng']
            target_counts['Nhãn'] = target_counts['Trạng thái (default)'].map({0: '0: Bình thường', 1: '1: Gian lận/Rủi ro'})
            fig_target = px.bar(
                target_counts, x='Nhãn', y='Số lượng', 
                color='Nhãn', title="Phân phối lớp mục tiêu (Gian lận vs Bình thường)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
        # 2. Phân phối của biến đầu vào quan trọng X_1 (Liên tục)
        with row1_col2:
            if 'X_1' in df_raw.columns:
                fig_x1 = px.histogram(
                    df_raw, x='X_1', color=TARGET if TARGET in df_raw.columns else None,
                    marginal='box', title="Phân phối của thuộc tính X_1",
                    barmode='overlay', color_discrete_sequence=px.colors.qualitative.Safe
                )
                st.plotly_chart(fig_x1, use_container_width=True)
                
        # 3. Phân phối của thuộc tính X_2 (Liên tục)
        with row2_col1:
            if 'X_2' in df_raw.columns:
                fig_x2 = px.histogram(
                    df_raw, x='X_2', color=TARGET if TARGET in df_raw.columns else None,
                    marginal='violin', title="Phân phối của thuộc tính X_2",
                    barmode='overlay', color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_x2, use_container_width=True)
                
        # 4. Mối quan hệ tương quan giữa X_1 và X_2
        with row2_col2:
            if 'X_1' in df_raw.columns and 'X_2' in df_raw.columns:
                fig_scatter = px.scatter(
                    df_raw.sample(min(1000, len(df_raw))), x='X_1', y='X_2', 
                    color=TARGET, title="Biểu đồ phân tán X_1 vs X_2 (Mẫu 1000 dòng)",
                    opacity=0.7, color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
        # Hỗ trợ cấu hình động nếu người dùng muốn khám phá thêm nhiều biến khác
        st.divider()
        st.markdown("### 🔍 Tùy chọn phân tích chuyên sâu các biến khác")
        all_features = [col for col in df_raw.columns if col != TARGET]
        selected_features = st.multiselect(
            "Chọn các biến bổ sung để xem biểu đồ phân phối hình hộp (Boxplot):", 
            options=all_features, default=all_features[:3]
        )
        if selected_features:
            fig_melted = px.box(
                df_raw, y=selected_features, color=TARGET if TARGET in df_raw.columns else None,
                title="Biểu đồ phân phối đa biến (Boxplot)", barmode="group"
            )
            st.plotly_chart(fig_melted, use_container_width=True)
    else:
        st.warning("Tệp dữ liệu không chứa cột mục tiêu 'default' để tiến hành phân tích phân phối.")

# ------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ------------------------------------------
with tab3:
    st.subheader("🎯 Đánh giá hiệu năng mô hình toán học")
    
    # Điều phối: Kiểm tra nếu chưa bấm huấn luyện ở sidebar
    if 'evaluation_metrics' not in st.session_state:
        st.info("ℹ️ Vui lòng thiết lập tham số và nhấn nút **🚀 Huấn luyện mô hình** tại Sidebar để xem kết quả phân tích kiểm định.")
    else:
        metrics = st.session_state['evaluation_metrics']
        
        # 1. Trình bày các chỉ tiêu vô hướng qua st.metric
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
        m_col2.metric("Độ chuẩn xác (Precision)", f"{metrics['precision']:.4f}")
        m_col3.metric("Độ nhạy (Recall)", f"{metrics['recall']:.4f}")
        m_col4.metric("Điểm F1-Score", f"{metrics['f1']:.4f}")
        
        st.divider()
        
        col_left, col_right = st.columns(2)
        
        # 2. Ma trận nhầm lẫn (Confusion Matrix)
        with col_left:
            st.markdown("#### 📊 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng"),
                x=['0 (Bình thường)', '1 (Gian lận)'],
                y=['0 (Bình thường)', '1 (Gian lận)'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification report dạng bảng
            st.markdown("#### 📝 Báo cáo chi tiết lớp (Classification Report)")
            report_dict = classification_report(metrics['y_true'], metrics['y_pred'], output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)

        # 3. Biểu đồ ROC Curve và Độ quan trọng của biến
        with col_right:
            if metrics['y_prob'] is not None:
                st.markdown("#### 📈 Đường cong ROC (Receiver Operating Characteristic)")
                fpr, tpr, thresholds = roc_curve(metrics['y_true'], metrics['y_prob'])
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.4f})', line=dict(color='darkorange', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Dự đoán ngẫu nhiên', line=dict(color='navy', width=2, dash='dash')))
                fig_roc.update_layout(xaxis_title='Tỷ lệ Dương tính giả (FPR)', yaxis_title='Tỷ lệ Dương tính thật (TPR)', margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_roc, use_container_width=True)
                
            st.markdown("#### 🌲 Độ quan trọng của thuộc tính (Feature Importance)")
            feat_imp_df = pd.DataFrame({
                'Thuộc tính': FEATURES,
                'Độ quan trọng': metrics['features_importance']
            }).sort_values(by='Độ quan trọng', ascending=True)
            
            fig_imp = px.bar(feat_imp_df, x='Độ quan trọng', y='Thuộc tính', orientation='h', color='Độ quan trọng', color_continuous_scale='Viridis')
            fig_imp.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ------------------------------------------
with tab4:
    st.subheader("🔮 Hệ thống dự báo rủi ro giao dịch mới")
    
    # Kiểm tra trạng thái mô hình
    if 'trained_model' not in st.session_state or 'data_scaler' not in st.session_state:
        st.info("ℹ️ Vui lòng chạy huấn luyện mô hình ở mục Sidebar trước khi thực hiện chức năng dự báo.")
    else:
        model = st.session_state['trained_model']
        scaler = st.session_state['data_scaler']
        
        mode = st.radio(
            "Chọn phương thức nhập dữ liệu dự báo đầu vào:",
            options=["Chế độ 1: Nhập trực tiếp thủ công", "Chế độ 2: Tải file dữ liệu loạt lớn (Excel/CSV)"],
            horizontal=True
        )
        
        # ------------------------------------------
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP
        # ------------------------------------------
        if mode == "Chế độ 1: Nhập trực tiếp thủ công":
            st.markdown("#### 📝 Điền thông số giao dịch cần kiểm tra")
            
            # Tính toán thông số cơ bản từ tập dữ liệu thô phục vụ giá trị mặc định lý tưởng
            with st.form("single_prediction_form"):
                form_cols = st.columns(3)
                input_data = {}
                
                for idx, feature in enumerate(FEATURES):
                    # Phân bổ đều các widget đầu vào vào 3 cột thiết kế
                    col_target = form_cols[idx % 3]
                    
                    # Lấy min, max, median từ dữ liệu gốc để cấu hình biên hợp lý
                    if feature in df_raw.columns:
                        min_val = float(df_raw[feature].min())
                        max_val = float(df_raw[feature].max())
                        median_val = float(df_raw[feature].median())
                    else:
                        min_val, max_val, median_val = -10.0, 10.0, 0.0
                        
                    input_data[feature] = col_target.number_input(
                        f"Giá trị biến {feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        format="%.4f",
                        help=f"Nhập thông số cho biến {feature}. Khoảng dữ liệu gốc: [{min_val:.2f} -> {max_val:.2f}]"
                    )
                
                submit_predict = st.form_submit_button("🛡️ Kiểm tra mức độ rủi ro", type="primary", use_container_width=True)
                
            if submit_predict:
                # Chuyển dữ liệu sang DataFrame đơn dòng
                input_df = pd.DataFrame([input_data])
                
                # Thực hiện chuẩn hóa chuẩn xác theo bộ scaler gốc
                input_scaled = scaler.transform(input_df)
                
                # SỬA LỖI WARNING: Khôi phục tên cột cho dữ liệu vừa được scale
                input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURES)
                
                # Thực hiện dự báo nhãn và xác suất sử dụng DataFrame đã sửa lỗi
                prediction = model.predict(input_scaled_df)[0]
                prediction_proba = model.predict_proba(input_scaled_df)[0]
                
                st.divider()
                st.markdown("### Kết quả đánh giá từ hệ thống AI:")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if prediction == 1:
                        st.error("⚠️ CẢNH BÁO: Giao dịch có dấu hiệu rủi ro cao / Gian lận!")
                    else:
                        st.success("✅ AN TOÀN: Giao dịch bình thường, độ tin cậy đạt yêu cầu.")
                        
                with res_col2:
                    st.metric(
                        label="Xác suất rủi ro (Probability of Fraud)", 
                        value=f"{prediction_proba[1] * 100:.2f} %",
                        delta=f"{'+' if prediction == 1 else ''}{(prediction_proba[1] - 0.5)*100:.1f}% so với ngưỡng an toàn"
                    )

        # ------------------------------------------
        # CHẾ ĐỘ 2 — TẢI FILE THEO CẤU TRÚC (Ví dụ X_new.xlsx)
        # ------------------------------------------
        else:
            st.markdown("#### 📂 Dự báo hàng loạt theo danh sách dữ liệu mới")
            st.caption("Hãy tải lên file mẫu chứa đầy đủ cấu trúc các cột đặc trưng từ `X_1` tới `X_14` (Không cần chứa cột `default`).")
            
            batch_file = st.file_uploader(
                "Tải lên tệp danh sách cần chấm điểm định kỳ", 
                type=["csv", "xlsx"], 
                key="batch_uploader"
            )
            
            if batch_file is not None:
                # Đọc tệp dữ liệu loạt mới
                batch_bytes = batch_file.getvalue()
                df_batch = load_data(batch_bytes, batch_file.name)
                
                if df_batch is not None:
                    # Kiểm tra tính hợp lệ của lược đồ cột (schema)
                    missing_batch_cols = [col for col in FEATURES if col not in df_batch.columns]
                    
                    if missing_batch_cols:
                        st.error(f"❌ File dữ liệu không khớp cấu trúc. Thiếu các cột đặc trưng sau: {missing_batch_cols}")
                    else:
                        # Trích xuất chính xác tập biến đưa vào mô hình dự đoán
                        X_batch = df_batch[FEATURES]
                        
                        # Sử dụng bộ tiền xử lý chuẩn hóa lúc train
                        X_batch_scaled = scaler.transform(X_batch)
                        
                        # SỬA LỖI WARNING: Chuyển đổi dữ liệu sau scale thành DataFrame kèm đúng tên cột
                        X_batch_scaled_df = pd.DataFrame(X_batch_scaled, columns=FEATURES)
                        
                        # Dự báo hàng loạt sử dụng DataFrame có tên cột
                        batch_predictions = model.predict(X_batch_scaled_df)
                        batch_proba = model.predict_proba(X_batch_scaled_df)[:, 1]
                        
                        # Ghép kết quả đầu ra vào DataFrame hiển thị
                        df_result = df_batch.copy()
                        df_result['Dự báo rủi ro (Prediction)'] = batch_predictions
                        df_result['Xác suất gian lận (Probability)'] = batch_proba
                        df_result['Kết luận'] = df_result['Dự báo rủi ro (Prediction)'].map({0: 'Bình thường', 1: 'Rủi ro / Gian lận'})
                        
                        st.success(f"⚡ Đã xử lý và chấm điểm thành công cho toàn bộ {len(df_result)} giao dịch!")
                        
                        # Thống kê tổng quan phân phối kết quả dự đoán hàng loạt
                        cnt_fraud = int((batch_predictions == 1).sum())
                        st.metric("Phát hiện tổng số ca rủi ro", f"{cnt_fraud} / {len(df_result)} giao dịch", 
                                  delta=f"Tỷ lệ {cnt_fraud/len(df_result)*100:.2f}%", delta_color="inverse")
                        
                        # Hiển thị bảng kết quả tổng hợp trong khung cuộn gọn gàng
                        st.markdown("### 📋 Bảng kết quả dự báo chi tiết:")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Xuất dữ liệu tải về ở định dạng CSV tiện ích (utf-8-sig để hiển thị tốt tiếng Việt trong Excel)
                        csv_buffer = io.StringIO()
                        df_result.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải về báo cáo kết quả dự báo (.CSV)",
                            data=csv_data,
                            file_name="Ket_qua_du_bao_gian_lan_hang_loat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
