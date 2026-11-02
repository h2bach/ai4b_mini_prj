# Checklist bài nộp

Áp dụng riêng mỗi project; demo hai bài tổng cộng tối đa 5 phút.

- [ ] Đọc đúng đề và dùng đúng split/seed, không sửa benchmark.
- [ ] Mean/std, TF-IDF, vocabulary và class weights chỉ học từ train.
- [ ] Chạy đủ majority và baseline ML, không còn cảnh báo chưa hội tụ.
- [ ] TODO model, forward và train tự hoàn thiện; giải thích được code.
- [ ] Hai cấu hình DL cùng seed 0 và budget, chỉ đổi biến đề yêu cầu.
- [ ] Shape và parameter tính tay khớp code.
- [ ] Kiểm loss/gradient hữu hạn, weight thay đổi và model học được trên tập nhỏ.
- [ ] History, best checkpoint, last checkpoint cho resume được lưu.
- [ ] Ít nhất 5 mẫu sai validation có ID và giả thuyết.
- [ ] Quy tắc đăng ký trước train; lua_chon.json lưu trước khi mở test.
- [ ] Test chỉ dùng lượt báo cáo cuối cho cấu hình đã khóa, không tune lại bằng test.
- [ ] Accuracy có số đúng/tổng; có macro-F1, F1/recall từng lớp và confusion matrix.
- [ ] Load checkpoint vào model mới bằng CPU, inference hoạt động.
- [ ] NLP: vocab/preprocessing/label mapping khớp checkpoint; input rỗng, UNK và padding được kiểm.
- [ ] CNN: demo đúng ảnh uint8 28×28 ngoài train; nêu giới hạn với ảnh điện thoại.
- [ ] Restart Kernel → Run All thành công, không phụ thuộc biến từ phiên cũ.
- [ ] Báo cáo ≤2 trang/bài, không chép điểm baseline chưa tự chạy làm kết quả.
- [ ] Nộp notebook có output, checkpoint, vocab (NLP), lua_chon.json, history/CSV, báo cáo.
- [ ] Không commit data/, runs/, checkpoint hoặc thông tin đăng nhập lên repo dùng chung.

Rubric mỗi bài: dữ liệu/đánh giá 25; baseline/thí nghiệm 20; model/shape/parameter 20; phân tích 20; tái hiện/báo cáo/demo 15. Điểm cuối là trung bình hai bài. DL thua ML vẫn được chấm tốt nếu quy trình đúng và phân tích có căn cứ.
