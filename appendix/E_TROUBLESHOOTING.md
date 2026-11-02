# Appendix E — Sửa lỗi theo triệu chứng

Ghi nguyên traceback, ô notebook, shape/dtype/device và thao tác tái hiện. Sửa nguyên nhân trước khi chạy lại full budget. Không xóa kiểm tra để “cho chạy qua”.

| Triệu chứng | Kiểm trước | Cách xử lý |
|---|---|---|
| Không tìm thấy `shared` | Đã clone/giải nén cả repo chưa? CWD ở đâu? | Làm bootstrap trong [CLOUD.md](../CLOUD.md), `os.chdir` tới root có shared/ |
| Thiếu pyarrow/matplotlib | Package của đúng kernel | Chỉ cài package thiếu bằng `%pip`; restart kernel nếu cần |
| CUDA unavailable | Runtime có cấp GPU? | Bật GPU theo nền tảng; CPU vẫn chạy đúng; không công bố runtime CPU là GPU |
| CPU/GPU mismatch | Tất cả input, model và loss weights | Đưa model/input/weights cùng device; length cho packing trên CPU |
| Expected Long nhưng nhận Float | Label hoặc token ID | Dùng int64; ảnh/embedding activation dùng float32 |
| mat1/mat2 không nhân được | Shape sau pooling và Flatten | Đối chiếu bảng shape; không thêm reshape tùy tiện |
| Index out of range Embedding | max ID và vocab_size | ID <V; load đúng vocab của checkpoint; UNK=1 |
| Length=0 | Input rỗng sau normalize | Báo ValueError rõ ràng; không âm thầm xóa dòng corpus |
| Predict đổi khi thêm PAD | Lấy hidden tại vị trí nào? | Dùng length thật, packed sequence và hidden cuối |
| Loss NaN/Inf | Input, gradient, learning rate | Kiểm finite từng bước; clipping sau backward; giữ lr theo đề |
| Weight không đổi | backward, step, parameter instance | Kiểm optimizer giữ weight của model đang train; không detach logits |
| Gradient tăng qua batch | zero_grad | Xóa gradient trước batch mới |
| Train tốt, validation kém | Overfit hoặc bug split/preprocess | Kiểm train-only stats và eval mode; không xem test để sửa |
| Baseline chưa hội tụ | ConvergenceWarning | Dừng, báo giảng viên, kiểm dtype/scaling; không lấy điểm run lỗi làm baseline |
| Resume khác config | last.pt, fingerprint, budget | Dùng đúng folder/config cũ hoặc folder thí nghiệm mới |
| Checksum khác | File tải dở hoặc nguồn/cache thay đổi | Tải lại file lỗi trong cache mới, so revision; không sửa expected hash cho qua |
| Mất output Colab | Output còn ở VM tạm? | Tải/chép checkpoint sau mỗi phiên; restore cùng config trước resume |
| OOM | Model cũ, tensor còn giữ graph | Restart kernel, xóa tensor không cần; không tự giảm batch riêng một cấu hình benchmark |

## Quy trình debug ngắn

1. Một batch: in shape, dtype, min/max, device; kiểm label đúng range.
2. Forward: logits đúng shape và hữu hạn.
3. Một update: gradient hữu hạn và ít nhất một weight thay đổi.
4. Tập train nhỏ đủ lớp: cố overfit để kiểm model học được. Ghi output riêng, không báo benchmark.
5. Một epoch subset: save, reload CPU, predict, kiểm PAD/UNK nếu NLP.
6. Chỉ sau đó train full, giữ cấu hình theo đề và lưu history.

`NotImplementedError` khi bật RUN_DL là lời nhắc TODO, không phải lỗi thư viện. Khi RUN_DL=False, baseline chạy bình thường, phần DL in hướng dẫn và không chạy lời giải thay thế.

## Khi cần gửi giảng viên

Gửi tên project, seed/variant, môi trường, traceback đầy đủ và ô tái hiện tối thiểu. Với kết quả khác baseline, gửi manifest fingerprint và bảng validation. Không chỉ gửi ảnh chụp “điểm thấp”.
