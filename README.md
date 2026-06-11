# 🛡️ Ứng dụng Phát hiện Giao dịch Gian lận & Rủi ro Tín dụng (Streamlit App)

Ứng dụng web tương tác thông minh được chuyển đổi từ mô hình huấn luyện Machine Learning (`phat_hien_giao_dich_gian_lan.ipynb`) nhằm tự động hóa quy trình phân tích dữ liệu, đánh giá kiểm định thuật toán và triển khai dự báo điểm rủi ro giao dịch trực tuyến theo thời gian thực hoặc theo chuỗi tập tin hàng loạt.

## ✨ Tính năng chính của hệ thống
- **Cấu hình động tham số AI:** Cho phép tùy biến trực quan các siêu tham số của thuật toán `RandomForestClassifier` (Số cây, Độ sâu tối đa, Tiêu chuẩn phân tách cây) ngay trên giao diện Sidebar.
- **Phân tích Tổng quan & Trực quan hóa:** Tự động khám phá cấu trúc dữ liệu thô, phân tích phân phối các lớp mục tiêu bất cân bằng (`default`) thông qua các biểu đồ động chuyên sâu thiết kế bởi Plotly.
- **Đánh giá kiểm định mô hình độc lập:** Tái hiện chính xác các thang đo hiệu năng cao cấp bao gồm: Ma trận nhầm lẫn (Confusion Matrix), Báo cáo lớp chi tiết (Classification Report), và Đường cong ROC-AUC.
- **Hai chế độ dự báo nghiệp vụ linh hoạt:**
  - *Nhập trực tiếp thủ công:* Điền nhanh thông số giao dịch riêng lẻ để hệ thống trả kết quả phân tích mức rủi ro tức thì đi kèm xác suất cụ thể.
  - *Xử lý tập tin hàng loạt:* Tải lên tệp chứa danh sách nhiều giao dịch mới cần kiểm tra (tương tự định dạng `X_new.xlsx`) để xuất file báo cáo chấm điểm hoàn chỉnh.

## 🛠️ Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo máy tính của bạn đã được cài đặt môi trường Python (Khuyến nghị phiên bản `Python 3.9` đến `3.12`).

### Bước 2: Cài đặt các thư viện phụ thuộc bắt buộc
Di chuyển thư mục làm việc đến nơi chứa bộ mã nguồn ứng dụng và chạy câu lệnh Terminal sau để tự động cài đặt:
```bash
pip install -r requirements.txt
