# AI4B — Hai mini projects CNN và LSTM

Mỗi cá nhân hoàn thành **cả hai bài trong 14 ngày**. Baseline Dummy và ML cổ điển đã viết đầy đủ; CNN/LSTM có TODO để bạn tự hoàn thiện. Không bắt buộc DL thắng ML: điểm dựa trên quy trình, thí nghiệm công bằng và cách giải thích.

| Bài | Đề | Notebook | Thí nghiệm bắt buộc |
|---|---|---|---|
| 1. CNN Fashion-MNIST | [Đọc đề CNN](cnn/DE_BAI.md) | [Mở notebook CNN](cnn/cnn_project.ipynb) | Flatten so với GAP, seed 0 |
| 2. LSTM UIT-VSFC | [Đọc đề NLP](nlp/DE_BAI.md) | [Mở notebook NLP](nlp/nlp_project.ipynb) | CE thường so với CE có trọng số, seed 0 |

## Bắt đầu

1. Đọc [hướng dẫn Kaggle/Colab](CLOUD.md), lấy **toàn bộ repo**, không chỉ file notebook.
2. Học [cầu nối NumPy → PyTorch 45 phút](lessons/01_numpy_to_pytorch.ipynb).
3. Mở notebook CNN, chạy baseline trước. Điền TODO và chỉ sau đó đặt `RUN_DL=True`.
4. Học [bổ trợ NLP 60 phút](lessons/02_nlp_foundations.ipynb), rồi làm project NLP.
5. Dùng [Appendix](appendix/README.md) để tra cứu shape, autograd, metric, vocabulary, LSTM và sửa lỗi.
6. Hoàn thiện [báo cáo](student/REPORT_TEMPLATE.md) và [checklist](student/SUBMISSION_CHECKLIST.md).

Chạy local nếu đã có Python/PyTorch phù hợp:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ai4b-student --display-name "AI4B Student"
```

Mở notebook bằng VS Code/Jupyter với kernel vừa tạo. PyTorch GPU cần bản phù hợp driver; cloud thường đã có torch, chỉ cài package thiếu theo CLOUD.md. Tuyến bài bắt buộc không dùng torchvision; helper đọc IDX và parquet trực tiếp.

## Lịch 14 ngày

| Ngày | Làm gì? | Bằng chứng |
|---|---|---|
| 1–2 | Môi trường, cầu nối, baseline CNN | ML chạy được; hiểu một update |
| 3–4 | TODO CNN và train | Shape/parameter, history, checkpoint |
| 5–6 | GAP, phân tích lỗi | Bảng validation, ≥5 ảnh sai, kiểm tiến độ 1 |
| 7–8 | Bổ trợ NLP, baseline TF-IDF | Phân bố lớp, hiểu vocab/PAD |
| 9–10 | TODO chuỗi và LSTM | Train/save/reload, kiểm PAD/UNK |
| 11–12 | Class weighting, phân tích, khóa lựa chọn | Bảng validation, ≥5 câu sai, kiểm tiến độ 2 |
| 13 | Mở test sau khóa, tái hiện, báo cáo | Test report, accuracy đúng/tổng |
| 14 | Nộp và demo | Hai bài, tổng demo ≤5 phút |

Tổng thực hành dự kiến 12–16 giờ; đây là lịch học, không phải cam kết runtime. Giảng viên bố trí hỗ trợ riêng.

## Quy tắc quan trọng

Train-only cho mean/std, TF-IDF, vocabulary và class weights. Split cố định. Đăng ký quy tắc trước train, chọn trên validation, ghi `lua_chon.json`, rồi mới mở test. Giữ cờ test đóng cho tới ngày 13. Gói không có đáp án DL hoặc checkpoint tham chiếu; chưa điền TODO thì phần DL in hướng dẫn và không chạy thay.

Output mặc định `runs/my_project/`, cache `data/`; hai thư mục này không đưa lên Git. Lưu checkpoint và notebook ra khỏi runtime cloud trước khi kết thúc phiên. Notebook chứa template TODO riêng; nếu sửa `student/answers.py`, cần cập nhật ô tương ứng hoặc import lại module bằng chủ đích.

[Thông tin dữ liệu và trích dẫn](DATA_SOURCES.md). Không phát lại công khai corpus UIT-VSFC từ cache. Điểm mỗi project trên 100, điểm đợt project là trung bình hai bài; xem rubric trong từng đề.
