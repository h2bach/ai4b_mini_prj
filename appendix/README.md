# Appendix — Tài liệu hỗ trợ mini projects

Phụ lục dùng để tra cứu trong lúc làm bài. Mỗi phần gồm khái niệm, ví dụ nhỏ, lỗi thường gặp, câu tự kiểm và nguồn chính thức. Ví dụ bổ trợ dùng dữ liệu đồ chơi hoặc Digits; CNN/LSTM của bài chính vẫn cần bạn tự hoàn thiện.

| Khi bạn cần… | Mở tài liệu | Thực hành đi kèm |
|---|---|---|
| Chuyển từ NumPy sang autograd/DataLoader | [A — PyTorch](A_PYTORCH.md) | [Bài cầu nối 45 phút](../lessons/01_numpy_to_pytorch.ipynb) |
| Tính shape, parameter, hiểu Flatten/GAP | [B — CNN](B_CNN_SHAPES.md) | Bảng shape trong [đề CNN](../cnn/DE_BAI.md) |
| Hiểu token, vocabulary, embedding, LSTM, PAD | [C — NLP](C_NLP.md) | [Bài bổ trợ 60 phút](../lessons/02_nlp_foundations.ipynb) |
| Đọc F1, chọn checkpoint, chống leakage | [D — Thực nghiệm](D_EVALUATION.md) | Confusion matrix và lua_chon.json |
| Sửa lỗi môi trường/shape/GPU/checkpoint | [E — Sửa lỗi](E_TROUBLESHOOTING.md) | [Hướng dẫn cloud](../CLOUD.md) |
| Tìm tài liệu học thêm và thuật ngữ | [F — Nguồn và thuật ngữ](F_RESOURCES.md) | Câu hỏi tự kiểm cuối mỗi phần |

Lộ trình gợi ý: A → CNN + B → C → NLP → D để chốt báo cáo. E dùng khi gặp lỗi; F dành cho phần muốn đọc sâu hơn. Không cần học hết phụ lục trước khi mở notebook.
