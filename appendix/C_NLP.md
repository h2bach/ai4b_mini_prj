# Appendix C — Từ câu văn đến LSTM

## 1. Normalization và tokenization

Unicode NFC đưa các cách mã hóa tương đương về một biểu diễn chuẩn. Lowercase giảm khác biệt hoa/thường. Tách khoảng trắng đơn giản: “giảng viên nhiệt tình” thành 4 token âm tiết. Đây không phải phân đoạn từ; “giảng viên” có thể là một đơn vị nghĩa gồm hai token.

Giữ nguyên quy tắc giữa train, validation, test và demo. Không fit lại vocabulary ở inference. Dấu câu được giữ trong token theo dữ liệu gốc; không tự thêm bước bỏ dấu/bỏ stopword trong thí nghiệm bắt buộc.

Tự kiểm: nếu train dùng NFC còn demo không dùng, cùng một chữ có thể thành UNK vì sao?

## 2. Vocabulary, PAD và UNK

Vocabulary là ánh xạ token→ID học từ train. ID không hàm ý thứ tự cảm xúc. Trong bài: PAD=0, UNK=1; giữ token xuất hiện ≥2; sort theo tần suất giảm, hòa theo từ điển. Quy tắc sắp xếp giúp tái hiện ID.

Ví dụ train đồ chơi: `['hay hay rõ', 'rõ dễ', 'lạ']`. Hãy tự đếm và lập vocabulary. Sau đó mã hóa câu “hay mới”: token nào là UNK? Pad đến 5 vị trí, length thật là bao nhiêu?

PAD biểu thị vị trí thêm để ghép batch; UNK biểu thị một token thật chưa biết. Một câu toàn token lạ vẫn có length dương. Một câu trống phải báo lỗi, vì packed sequence không nhận chuỗi dài 0.

Tỷ lệ UNK validation = số token validation không thuộc vocab / tổng token validation. Nêu rõ thống kê tính trước hay sau cắt; summary của runner dùng toàn bộ token trước cắt. Báo thêm tỷ lệ câu >40 token để thấy mức mất thông tin.

## 3. Embedding là bảng tra

E có shape `(V,d)`. Token ID i chọn hàng E[i], tạo vector d số thực được học. One-hot(i) nhân E cho cùng vector. Embedding `(N,L,d)` không đồng nghĩa model đã hiểu ngữ nghĩa; weight cần được train bằng task.

`padding_idx=0` chặn update từ PAD qua cơ chế Embedding; vẫn cần length/packing để không lặp LSTM qua PAD. Parameter embedding là V×d, tính cả hàng PAD trong số lượng parameter.

Tự kiểm: thay ID mapping nhưng giữ nguyên weight embedding cũ có giữ hành vi model không? Vì sao checkpoint phải lưu vocab?

## 4. RNN và LSTM

RNN cập nhật hidden theo token và hidden trước đó. Vì thế thứ tự có ảnh hưởng; “không hay” khác “hay không”. Lặp qua chuỗi dài có thể làm gradient nhỏ hoặc lớn.

LSTM giữ cell state c và hidden h, dùng các cổng để điều tiết:

- `f_t`: phần cell cũ cần giữ.
- `i_t` và `g_t`: mức và nội dung mới cần ghi.
- `c_t = f_t*c_(t-1) + i_t*g_t`.
- `o_t`: mức đưa cell ra hidden; `h_t = o_t*tanh(c_t)`.

Các tích ở đây theo từng phần tử. Trong bài chính, mỗi câu nhận một nhãn: many-to-one. Bạn dùng hidden cuối của chuỗi thật để đưa vào head; không dùng cell state thay hidden mà không giải thích.

Tính parameter LSTM một lớp: bốn cổng, mỗi cổng có weight input `h*d`, weight recurrent `h*h`, và hai bias h của PyTorch. Tổng `4h(d+h+2)`. Với d=h=64, cộng embedding và Linear64→3 để tự suy ra `64V+33.475`.

## 5. Length, packing và h_n

Hai câu có length 2 và 5 cùng pad đến 40. Nếu chạy không packing rồi lấy output ở vị trí 39, model đã đi qua nhiều bước PAD khác nhau. `pack_padded_sequence` dùng length thật để tránh các bước thừa. `enforce_sorted=False` cho phép batch không sort trước.

Với một lớp một chiều, `h_n` có shape `(1,N,64)`. `h_n[-1]` là `(N,64)`, phù hợp cho Linear64→3. Length phải là số nguyên, dương, không vượt chiều sequence; truyền length trên CPU khi pack.

Kiểm bắt buộc: cùng một câu chạy riêng và trong batch với câu dài hơn phải cho logits gần bằng nhau (`atol=1e-6`, `rtol=1e-5`). Kiểm này phát hiện cả lỗi PAD và lỗi thứ tự batch.

## 6. Class weighting và clipping khác nhau

Class weighting tăng tác động tương đối của lớp hiếm trong loss. Dùng `N/(3*N_c)` chỉ từ train; không dùng số lượng validation để tính. Loss có trọng số có thể tăng recall neutral nhưng tăng false positive của neutral; kết luận bằng metric thực.

Clipping giới hạn norm gradient sau backward, trước step. Nó kiểm soát độ lớn update gián tiếp; không xử lý mất cân bằng lớp và không tạo dữ liệu mới. Hai biến thể LSTM đều dùng clipping 1.0.

Nguồn: [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html), [Packing](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html), [Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html), [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback).
