# Appendix F — Nguồn học và thuật ngữ

## Nguồn đọc theo mục đích

| Nguồn chính thức | Nên đọc để làm gì? | Câu hỏi sau khi đọc |
|---|---|---|
| [PyTorch Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Tensor, dataset, model, autograd, optimizer | Bạn ghép được các phần thành một epoch chưa? |
| [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) | Hiểu mean/std áp cho cả batch | Vì sao scalar mean chuẩn hóa được nhiều ảnh? |
| [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) | Shape, padding và weight | Bạn tính được parameter cả bias chưa? |
| [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | Logits, label index, class weight | Vì sao không softmax trước loss? |
| [Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) | Bảng tra token và padding_idx | Vì sao input là ID int64? |
| [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) | Input/output, h_n, c_n, bias | Shape hidden cuối với một lớp một chiều là gì? |
| [Packing](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) | Length thật và batch khác độ dài | Padding có được tính như token thật không? |
| [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) | N-gram và fit/transform | Tập nào được fit vectorizer? |
| [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | C, solver và class_weight | C nhỏ hơn thay regularization thế nào? |
| [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) | Macro và các cách gộp khác | Macro khác accuracy trong bộ lệch lớp ra sao? |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | Dữ liệu ảnh và thứ tự 10 label | Ảnh này khác ảnh điện thoại ở đâu? |
| [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) | Corpus, mapping label, nguồn | Sentiment khác topic ở đâu? |
| [Colab FAQ](https://research.google.com/colaboratory/faq.html) | Runtime tạm, GPU và giới hạn tài nguyên | Bạn đã lấy checkpoint khỏi VM chưa? |
| [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) | Notebook, input/output và môi trường | Output cần lưu ở thư mục nào? |

Các liên kết `stable` có thể thay đổi theo phiên bản; báo cáo của bạn phải ghi phiên bản đã chạy. Không cập nhật thư viện chỉ để khớp số benchmark của người khác.

## Từ điển ngắn

| Thuật ngữ | Nghĩa trong bài |
|---|---|
| Batch | Nhóm mẫu dùng trong một lần forward/update |
| Epoch | Một lượt qua toàn bộ train |
| Logit | Điểm trước softmax của một lớp |
| Gradient | Đạo hàm loss theo parameter |
| Parameter | Weight/bias được model quản lý và học |
| Hyperparameter | Lựa chọn như lr, batch, số epoch |
| Checkpoint | Snapshot model và metadata; last còn có trạng thái resume |
| Manifest | Danh sách mẫu/split và dấu vết nguồn để tái hiện |
| Fingerprint | Hash mô tả dữ liệu/cấu hình để phát hiện lệch |
| Vocabulary | Ánh xạ token→ID chỉ xây từ train |
| PAD | Vị trí đệm không phải token thật |
| UNK | Token thật chưa có trong vocabulary |
| Hidden state | Trạng thái chuỗi tại một bước |
| Macro-F1 | Trung bình F1 các lớp, mỗi lớp trọng số bằng nhau |
| Ablation | Thay một thành phần để đo tác động |
| Leakage | Dùng thông tin ngoài phạm vi train/selection cho phép |
| Domain shift | Dữ liệu mới khác phân bố dữ liệu đã kiểm |
| Baseline | Mốc so sánh đơn giản, chạy được và được đo trên cùng split |

## Tự kiểm trước demo

Không cần mở code, hãy giải thích trong 3 phút: dữ liệu chia thế nào; phần nào fit train; macro-F1 đo gì; một batch train làm gì; checkpoint nào được chọn; thí nghiệm đổi gì; một lỗi thật; một giới hạn. Nếu chưa giải thích được, quay lại phần tương ứng thay vì chỉ chạy thêm epoch.
