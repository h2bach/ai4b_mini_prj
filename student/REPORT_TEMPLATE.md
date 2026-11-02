# Báo cáo mini project — tối đa 2 trang/bài

Họ tên / mã học viên: … · Project: … · Ngày: … · Python/PyTorch/sklearn: … · Thiết bị: …

## 1. Câu hỏi và dữ liệu

Câu hỏi của thí nghiệm: …

Nguồn/revision/fingerprint: … · Số train/validation/test: … · Mapping label: …

Preprocessing fit trên train gồm: … · Kiểm trùng ID/nội dung, mẫu rỗng: …

Riêng NLP: V=…, max length=…, tỷ lệ cắt=…, tỷ lệ UNK validation=… (định nghĩa mẫu số).

## 2. Kiến trúc và cách chọn

| Lớp | Input/output shape | Công thức parameter | Số parameter |
|---|---|---|---:|
| … | … | … | … |

Tổng tính tay / tổng code: … / …

Quy tắc chọn đăng ký trước train: … · Điều kiện giữ nguyên: … · Seed: …

Checkpoint đã chọn / best epoch / hash: … · Thời điểm khóa trước test: …

## 3. Kết quả đo thực tế

| Model/cấu hình | Seed | Params | Validation macro-F1 | Val accuracy (đúng/tổng) | F1 lớp quan tâm | Train/fit giây |
|---|---:|---:|---:|---|---:|---:|
| Majority | — | — | … | … | … | … |
| Logistic regression / TF-IDF | 0 | — | … | … | … | … |
| Flatten / plain | 0 | … | … | … | … | … |
| GAP / weighted | 0 | … | … | … | … | … |

Test sau khóa: bảng cho baseline và cấu hình đã chọn, ghi macro-F1, accuracy đúng/tổng, F1/recall từng lớp. Đính kèm CSV/confusion matrix thay vì nhét toàn bộ output vào 2 trang.

So sánh chất lượng/số parameter/runtime: … · Lớp nào được lợi/thiệt: …

## 4. Năm lỗi validation và diễn giải

| Sample ID | Nhãn thật → dự đoán | Quan sát cụ thể | Giả thuyết/cách kiểm |
|---|---|---|---|
| 1 … | … | … | … |
| 2 … | … | … | … |
| 3 … | … | … | … |
| 4 … | … | … | … |
| 5 … | … | … | … |

## 5. Kết luận và giới hạn

Trả lời câu hỏi bằng số đo: …

Một giới hạn (một seed, domain shift, corpus ít neutral, dữ liệu trùng, nhãn mơ hồ…): …

Cách chạy lại và demo sau reload CPU: … · Đường dẫn notebook/checkpoint/lua_chon.json: …
