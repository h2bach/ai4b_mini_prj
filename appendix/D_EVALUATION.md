# Appendix D — Metric và thực nghiệm có thể giải thích

## 1. Ba tập có ba vai trò

Train dùng để học weight và fit mean/std, TF-IDF, vocabulary, class weights. Validation dùng để chọn epoch/cấu hình và phân tích lỗi. Test đo kết quả sau khi mọi lựa chọn đã khóa. Đổi config vì xem test làm test trở thành tập chọn và mất ý nghĩa đánh giá cuối.

Ví dụ leakage: thấy một từ chỉ có ở validation rồi thêm nó vào vocab để giảm UNK. Dù không dùng label validation, bạn đã dùng phân bố ngoài train để thay preprocessing. Trong bài này vocabulary chỉ nhận danh sách train.

Trùng sample ID là lỗi chia tập. Trùng nội dung ở các ID khác nhau có thể là đặc tính dữ liệu; UIT-VSFC giữ split gốc và báo thống kê trùng, không tự xóa mẫu. Những giới hạn này cần nêu trong báo cáo.

## 2. Confusion matrix và macro-F1

Với một lớp c: TP là đúng c; FP là dự đoán c nhưng thật lớp khác; FN là thật c nhưng dự đoán khác.

`precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2TP/(2TP+FP+FN)`.

Macro-F1 lấy trung bình F1 của mọi lớp với trọng số bằng nhau. Trong helper, lớp không có dự đoán đúng và mẫu số 0 nhận 0. Accuracy là tổng đúng / tổng mẫu; phải ghi cả số đúng/tổng để người đọc biết quy mô.

Bài nhỏ: confusion matrix ba lớp, hàng thật/cột đoán: `[[45,0,5],[4,0,1],[5,0,40]]`. Tính accuracy, recall neutral và F1 từng lớp. Accuracy cao có đồng nghĩa model làm tốt neutral không?

Đừng nhầm macro-F1 với weighted-F1 hoặc micro-F1 khi đọc bài báo. Chỉ so số có cùng định nghĩa metric, dữ liệu và split.

## 3. Kiểm soát biến và seed

CNN chỉ đổi head; NLP chỉ đổi trọng số loss. Cùng seed giúp so sánh công bằng hơn nhưng không loại hết nhiễu. Khởi tạo lại model, optimizer và generator của DataLoader cho mỗi cấu hình. Không lấy model đã train làm điểm xuất phát của cấu hình còn lại.

Bài nộp bắt buộc seed 0 cho cả hai cấu hình. Nếu mở rộng chạy thêm nhiều seed, dùng cùng tập seed cho mỗi cấu hình và báo mean macro-F1 validation cùng population standard deviation (`ddof=0`). Nhiều seed giúp nhìn dao động; vài seed vẫn chưa đủ chứng minh khác biệt thống kê tổng quát. Không chọn seed cao nhất làm đại diện.

Thiết bị, phiên bản thư viện và phép toán có thể làm số khác nhau. Ghi môi trường, không cam kết giống từng bit giữa GPU khác nhau.

## 4. Best, last và khóa lựa chọn

`best.pt` được chọn bằng macro-F1 validation; hòa giữ epoch sớm hơn. `last.pt` chứa model + optimizer + RNG cuối epoch để resume cùng config. Đánh giá cuối dùng best, không tự chọn last vì biết test cao hơn.

`protocol.json` được đăng ký trước train. `lua_chon.json` chứa config được chọn, seed, metric validation, checkpoint và SHA-256. Sau khi khóa mới bật `OPEN_TEST`. Báo cáo test lưu riêng; chạy lại cùng khóa trả lại báo cáo đã tạo. Cờ test là hỗ trợ quy trình học, không phải cơ chế bảo mật giấu dữ liệu công khai.

Tự kiểm: thay weight nhưng giữ tên `best.pt` có đủ để xem là cùng model không? Vì sao lưu hash?

## 5. Viết phân tích có bằng chứng

Mỗi lỗi validation ghi ID, nhãn thật, dự đoán và quan sát cụ thể. Tách điều quan sát được (“câu có từ không”) với giả thuyết (“model bỏ qua phủ định”). Đề xuất cách kiểm: tìm thêm câu tương tự, xem UNK, so model khác trên cùng mẫu.

Không suy rộng từ 5 ví dụ thành tỷ lệ lỗi toàn corpus. Không tự sửa label để làm điểm đẹp. Nếu nghi nhãn mơ hồ, ghi nghi vấn và giới hạn.

Bảng kết quả dùng cùng split và metric. Báo cả kết quả kém; GAP hoặc LSTM thua vẫn có giá trị học. Không sửa kết luận để buộc DL thắng ML.

Nguồn: [F1 trong scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [Save/load](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).
