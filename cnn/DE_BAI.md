# Mini project 1 — CNN nhận diện quần áo

Mỗi học viên tự làm bài này và project LSTM. Ngân sách cả hai bài khoảng 12–16 giờ trong 14 ngày. Mở [notebook](cnn_project.ipynb), dùng [phụ lục](../appendix/README.md) khi cần tra cứu. Baseline ML đã hoàn chỉnh; phần CNN và train step do bạn tự viết.

## Câu hỏi và mục tiêu

CNN cải thiện những nhầm lẫn nào so với logistic regression trên pixel? Khi thay Flatten bằng Global Average Pooling (GAP), số parameter và chất lượng thay đổi thế nào?

Bạn cần đi trọn quy trình dữ liệu → baseline → CNN → validation → checkpoint → test → phân tích → demo. Không bắt buộc CNN thắng logistic regression. Bài làm tốt phải có bằng chứng và giải thích được lựa chọn.

## Dữ liệu và các điều kiện cố định

Fashion-MNIST gồm 70.000 ảnh xám 28×28, 10 lớp. Helper tải từ [repo Zalando](https://github.com/zalandoresearch/fashion-mnist), pin revision và xác minh checksum. Label giữ đúng thứ tự: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

| Thành phần | Quy định |
|---|---|
| Train | 50.000 ảnh tách từ train gốc |
| Validation | 10.000 ảnh, stratify, split_seed=42 |
| Test | 10.000 ảnh test gốc, chỉ mở sau khi khóa lựa chọn |
| Input | float32 `(N,1,28,28)` sau chuẩn hóa |
| Preprocessing | pixel/255; mean/std chỉ tính train, áp dụng cho mọi split và inference |
| Augmentation | Không dùng trong thí nghiệm bắt buộc |

Manifest dùng ID `train:i` và `test:i` để tránh nhầm hai nguồn index. Không chia lại giữa các cấu hình. EDA ban đầu chỉ xem train. Dữ liệu raw cache ở `data/`; không đưa vào Git.

## Phần 1 — Hiểu dữ liệu và baseline (ngày 1–2)

1. Chạy ô môi trường; ghi Python, PyTorch và CPU/GPU thực dùng.
2. Vẽ ít nhất một ảnh train mỗi lớp. Ghi shape, dtype, range trước/sau chuẩn hóa.
3. Giải thích tại sao chỉ tính mean/std từ train. Xác nhận ID các split không giao nhau.
4. Chạy majority và logistic regression có sẵn. Logistic regression flatten thành 784 feature, `C=0.1`, `solver=lbfgs`, `max_iter=1000`.
5. Ghi accuracy kèm đúng/tổng, macro-F1, recall/F1 từng lớp, confusion matrix, thời gian fit/predict. Cảnh báo chưa hội tụ phải được xử lý trước khi lấy số baseline chính thức.
6. Đọc confusion matrix và chọn hai cặp lớp thường nhầm; quan sát riêng Shirt. Viết giả thuyết từ ảnh train, chưa tune bằng test.

Kết quả cần thấy: baseline tạo được CSV có sample ID và metric validation. Nếu helper báo lỗi checksum hoặc split, dừng để kiểm cache; không xóa assertion.

## Phần 2 — Tự hoàn thiện CNN (ngày 3–4)

Điền TODO C1 (khai báo), C2 (forward) và T1 (train). Có thể sửa trực tiếp ô notebook hoặc `student/answers.py`; notebook chứa bản sao template, nên sửa module không tự thay ô notebook.

| Lớp | Output bỏ chiều batch | Parameter cần tự tính |
|---|---|---|
| Input | 1×28×28 | 0 |
| Conv(1→16, kernel3, padding1) | 16×28×28 | Tính cả bias |
| ReLU → MaxPool2 | 16×14×14 | 0 |
| Conv(16→32, kernel3, padding1) | 32×14×14 | Tính cả bias |
| ReLU → MaxPool2 | 32×7×7 | 0 |
| Flatten | 1568 | 0 |
| Linear(1568→10) | 10 logits | Tính cả bias |

Tổng parameter phải khớp **20.490**. Dùng [Appendix B](../appendix/B_CNN_SHAPES.md) để tính, không chỉ chép con số.

Cấu hình: Adam, lr=0.001, batch=128, 10 epoch, FP32, cross-entropy nhận logits trực tiếp. Train step phải có train mode, zero gradient, forward, loss, backward, kiểm gradient hữu hạn, optimizer step. Validation dùng eval mode và no_grad; không update weight.

Trước khi train đủ dữ liệu, kiểm shape `(N,10)`, parameter, weight thay đổi và model có khả năng overfit tập nhỏ đủ lớp. Chạy kiểm subset trong thư mục riêng; không ghi điểm subset vào bảng benchmark. Checkpoint tốt nhất có macro-F1 validation cao nhất; hòa lấy epoch sớm hơn. `last.pt` lưu cả optimizer và RNG để resume cùng cấu hình.

Kết quả cần thấy: loss có xu hướng giảm, weight thay đổi, history theo epoch, checkpoint load lại được. Nếu loss hữu hạn nhưng chất lượng không tăng, dùng [Appendix E](../appendix/E_TROUBLESHOOTING.md).

## Phần 3 — Thí nghiệm Flatten và GAP (ngày 5–6)

Chỉ thay head Flatten → Linear(1568,10) bằng GAP → Linear(32,10). Hai lớp convolution giữ nguyên. Model GAP phải có **5.130** parameter.

Giữ split, seed 0, hai lớp conv, Adam/lr, batch, budget 10 epoch và quy tắc chọn checkpoint. Khởi tạo lại từ seed cho mỗi cấu hình; không fine-tune từ model trước. Convolution khởi tạo giống nhau; số weight của hai head khác nhau. Không tăng budget riêng cho cấu hình có điểm thấp.

So sánh macro-F1, accuracy đúng/tổng, metric từng lớp, số parameter và thời gian train thực đo. Bàn luận:

- Giảm bao nhiêu parameter, theo số tuyệt đối và tỷ lệ?
- GAP giữ thông tin gì và làm mất thông tin vị trí nào?
- Shirt và ít nhất một lớp khác thay đổi thế nào?
- Có nên chọn model nhỏ hơn dựa trên số đo hiện có?
- Một seed có đủ để khẳng định phương pháp luôn tốt hơn không?

Phân tích ít nhất 5 ảnh sai validation theo ID; ưu tiên lỗi liên quan Shirt nếu quan sát được. Mỗi ảnh cần nhãn thật, dự đoán, giả thuyết và cách kiểm giả thuyết. Nếu không có 5 lỗi, báo đúng số lỗi và trao đổi với giảng viên, không tự tạo lỗi.

## Phần 4 — Chốt, test và demo (ngày 11–14)

Ghi quy tắc chọn trước khi train (notebook có `register_protocol`). Chọn theo macro-F1 validation, hòa ít parameter hơn, rồi thứ tự Flatten/GAP. Sau khi hoàn thiện phân tích, bật `LOCK_SELECTION` để tạo `lua_chon.json`; chỉ sau đó mới bật `OPEN_TEST`. Test chỉ dùng lượt báo cáo cuối. Chạy lại cùng checkpoint là tái hiện, không phải cơ hội tune tiếp.

Load checkpoint vào model mới bằng CPU rồi chọn một ảnh validation ngoài train để demo. Hiển thị ảnh, nhãn dự đoán và điểm cả 10 lớp. Không cần ảnh điện thoại. Phải nêu rõ kết quả Fashion-MNIST chưa chứng minh hiệu quả trên ảnh quần áo ngoài đời.

## Bài nộp và chấm điểm

Nộp notebook có output chạy từ kernel sạch, `best.pt`, `last.pt` nếu cần resume, `lua_chon.json`, preprocessing và label mapping (đã trong checkpoint), history/CSV và báo cáo tối đa 2 trang theo [mẫu](../student/REPORT_TEMPLATE.md). Có đủ hai baseline và hai cấu hình DL seed 0. Tổng demo cả hai project tối đa 5 phút.

| Tiêu chí | Điểm | Bằng chứng |
|---|---:|---|
| Dữ liệu, split, validation, test | 25 | Manifest, train-only normalization, khóa trước test |
| Baseline và thí nghiệm công bằng | 20 | Hai baseline, hai head, điều kiện giữ nguyên |
| CNN và giải thích shape/parameter | 20 | Code hoạt động, tính tay khớp |
| Phân tích lỗi, diễn giải | 20 | ≥5 ảnh sai validation, kết luận có giới hạn |
| Tái hiện, báo cáo, demo | 15 | Kernel sạch, checkpoint, đúng/tổng, demo |

Điểm cuối đợt project là trung bình hai bài. Accuracy cao nhất lớp không có điểm thưởng. Không đổi mô hình sau khi đã xem test.
