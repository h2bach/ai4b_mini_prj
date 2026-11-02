# Mini project 2 — LSTM đọc góp ý tiếng Việt

Mỗi học viên làm bài này cùng [project CNN](../cnn/DE_BAI.md). Mở [notebook](nlp_project.ipynb), học [bài bổ trợ NLP](../lessons/02_nlp_foundations.ipynb) và [phụ lục NLP](../appendix/C_NLP.md). TF-IDF + logistic regression được cung cấp hoàn chỉnh; biểu diễn chuỗi, LSTM và train step có TODO.

## Câu hỏi và mục tiêu

LSTM so với TF-IDF + logistic regression như thế nào trên góp ý sinh viên? Loss có trọng số tác động thế nào đến lớp neutral hiếm và hai lớp còn lại?

Kết quả hợp lệ có thể là LSTM thua TF-IDF. Mục tiêu là bạn tự làm được pipeline và giải thích bằng metric, mẫu sai và giới hạn của thí nghiệm.

## Dữ liệu, nguồn và mapping label

Dùng sentiment của [UIT-VSFC do nhóm UIT NLP công bố](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback), không dùng topic. Giữ nguyên 11.426 train / 1.583 validation / 3.166 test; label 0=negative, 1=neutral, 2=positive. Helper pin revision bản parquet chính thức, kiểm checksum và số mẫu. Không thực thi remote dataset script.

Đọc `data/nlp/manifest.json`: revision, SHA-256 từng file, ID và thống kê trùng nội dung sau NFC/lowercase. Sample ID không được giao nhau. Câu trùng nội dung giữa split phải được báo cáo; không tự đổi split hoặc xóa mẫu để cải thiện điểm.

Trích dẫn: Nguyen et al. (2018), *UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis*, KSE, pp. 19–24, DOI: [10.1109/KSE.2018.8573337](https://doi.org/10.1109/KSE.2018.8573337). [Trang tác giả](https://github.com/kietnv/uit-vsfc). Thẻ Hugging Face ghi license `unknown`; repo bài tập chứa helper tải và trích dẫn, không chứa bản sao corpus để phát lại công khai.

## Phần 1 — Dữ liệu và baseline (ngày 7–8)

1. Đọc ít nhất ba câu train mỗi lớp; giải thích negative, neutral, positive trong ngữ cảnh góp ý học tập.
2. Đếm lớp, chiều dài câu, tỷ lệ >40 token và mẫu rỗng/không hợp lệ. Nếu dữ liệu có mẫu rỗng, helper báo split/dòng; không âm thầm bỏ mẫu.
3. Chạy majority (lớp nhiều nhất chỉ tính train).
4. Chạy NFC/lowercase → TF-IDF unigram/bigram, `min_df=2` → logistic regression `C=1`, `solver=lbfgs`, `class_weight=balanced`, `max_iter=1000`. Vectorizer chỉ fit train.
5. Ghi macro-F1, F1/recall mỗi lớp, accuracy đúng/tổng, confusion matrix, thời gian fit/predict. Nêu vì sao accuracy có thể che neutral.

Kết quả cần thấy: phần ML chạy được trước khi điền TODO. Baseline có CSV sample ID validation; test chưa được đưa vào pipeline fit hoặc chọn.

## Phần 2 — Văn bản thành tensor (ngày 9)

Điền TODO N1 (`build_vocab`) và N2 (`encode_text`):

1. Unicode NFC → lowercase → tách khoảng trắng. Đây là token âm tiết, không giả định đã phân đoạn từ tiếng Việt.
2. Đếm token chỉ trên train. Giữ tần suất ≥2. `PAD=0`, `UNK=1`.
3. Sort tần suất giảm dần; hòa theo thứ tự từ điển. Gán ID từ 2. Không hardcode vocabulary size.
4. Cắt tối đa 40 token; token lạ thành UNK; pad bằng PAD; lưu chiều dài thật sau cắt, không tính PAD.
5. Trả ID int64 `(batch,40)`, length int64 `(batch,)`, label int64 `(batch,)`. Input rỗng phải có thông báo `ValueError` rõ ràng.

Theo dấu “giảng viên nhiệt tình” qua normalization, token, ID, PAD, length và embedding. Đo tỷ lệ UNK trên validation (tử số và mẫu số đều tính trên token). Giải thích tại sao không thêm token validation vào vocabulary.

## Phần 3 — LSTM và train (ngày 9–10)

Điền TODO N3/N4: Embedding(V,64,padding_idx=0) → LSTM(input64,hidden64,1 layer,1 chiều) → Linear(64,3). Trả ba logits, không softmax trước cross-entropy.

Dùng `pack_padded_sequence` với length trên CPU và `enforce_sorted=False`. Dùng `h_n[-1]` để biểu diễn hidden sau token thật cuối; không lấy output tại PAD cuối. Khung packed sequence có trong notebook, bạn hoàn thiện các bước còn lại và giải thích lý do.

| Thành phần | Shape cần giải thích |
|---|---|
| Token IDs | `(N,40)` |
| Embedding | `(N,40,64)` |
| Hidden cuối | `(1,N,64)` trước lấy lớp cuối |
| Logits | `(N,3)` |

Tính parameter thành **64V + 33.475**, rồi đối chiếu `sum(p.numel()...)` với V thực tế. PAD row vẫn là parameter trong bảng embedding dù gradient của hàng này bị chặn.

Adam, lr=0.001, batch=128, **12 epoch**, FP32, cross-entropy; gradient clipping max norm=1.0 sau backward, trước optimizer step. Tái sử dụng TODO T1 từ CNN và bổ sung clipping. Kiểm gradient hữu hạn, weight thay đổi và thử overfit tập nhỏ đủ lớp trong output riêng. Lưu best theo macro-F1 validation, hòa lấy epoch sớm hơn; lưu last để resume cùng RNG/optimizer.

## Phần 4 — Thí nghiệm class weighting (ngày 11–12)

Chạy seed 0 hai lần khởi tạo giống nhau: CE thường và CE có trọng số `w_c=N_train/(3*N_c)` tính từ **train**. Giữ vocabulary, kiến trúc, seed, thứ tự batch, optimizer/lr, batch size, 12 epoch và clipping giống nhau. Không train tiếp từ model của cấu hình trước.

Ghi macro-F1, accuracy đúng/tổng, F1/recall từng lớp, parameter, thời gian và best epoch. Không dùng độ lớn loss hai cấu hình để chọn model vì hàm mục tiêu khác nhau.

Phải trả lời:

- Neutral tăng/giảm bao nhiêu điểm F1 và recall? Có đánh đổi với negative/positive không?
- Accuracy và macro-F1 có thể đi theo hai hướng nào? Vì sao?
- LSTM có lợi ích gì so với TF-IDF trong số đo của bạn? Một seed giới hạn kết luận thế nào?
- Vì sao class weighting không tạo thêm ví dụ neutral mới?

Phân tích **ít nhất 5 câu sai validation** với ID, nhãn thật, dự đoán và giả thuyết. Kiểm tra phủ định, câu vừa khen vừa chê, neutral, UNK. Không ép đủ mỗi nhóm nếu không có lỗi thật; ghi rõ nhóm không quan sát được. Không dùng câu test để tune.

## Phần 5 — Khóa, test và demo (ngày 13–14)

Quy tắc chọn đăng ký trước train: macro-F1 validation; hòa ít parameter hơn (hai LSTM bằng nhau), rồi thứ tự plain/weighted. Tạo `lua_chon.json` trước khi bật `OPEN_TEST`. Báo cáo test một lượt cho baseline và checkpoint đã chọn. Chạy lại cùng checkpoint để tái hiện không cho phép quay về tune.

Load model mới bằng CPU cùng **vocabulary, preprocessing và label mapping trong checkpoint**. Kiểm một câu predict riêng và trong batch có nhiều PAD cho cùng kết quả trong sai số số học. Kiểm input rỗng báo lỗi và từ mới dùng UNK. Demo bằng câu tự gõ, hiển thị label và điểm cả ba lớp. Câu tự gõ chỉ quan sát hành vi, không được cộng vào benchmark.

## Nộp bài và rubric

Nộp notebook có output chạy từ kernel sạch, checkpoint có đủ vocab/preprocessing/labels, `vocab.json`, `lua_chon.json`, history/CSV và báo cáo tối đa 2 trang theo [mẫu](../student/REPORT_TEMPLATE.md). Có majority, TF-IDF và hai loss seed 0. Demo chung cả hai project ≤5 phút.

| Tiêu chí | Điểm | Bằng chứng |
|---|---:|---|
| Dữ liệu, split, validation, test | 25 | Vocabulary/TF-IDF fit train, mapping chuẩn, khóa trước test |
| Baseline và thí nghiệm công bằng | 20 | Hai baseline; hai loss giữ nguyên điều kiện |
| LSTM và giải thích | 20 | Shape/parameter, length, PAD, gradient clipping |
| Phân tích lỗi và diễn giải | 20 | ≥5 lỗi validation; phân tích neutral, đánh đổi và giới hạn |
| Tái hiện, báo cáo, demo | 15 | Kernel sạch, save/reload CPU, UNK/rỗng, đúng/tổng |

Không cộng điểm vì đứng đầu accuracy. Điểm cuối đợt project là trung bình hai bài.
