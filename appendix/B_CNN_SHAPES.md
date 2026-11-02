# Appendix B — Shape, convolution và số parameter

## 1. Tính kích thước trước khi chạy

Cho một chiều input H, kernel K, padding P, stride S và dilation D:

`H_out = floor((H + 2P - D*(K-1) - 1)/S + 1)`.

Với 28, kernel3, padding1, stride1, dilation1: output vẫn 28. MaxPool2 stride2 biến 28→14; khối thứ hai biến 14→7. Số kênh output do số filter của Conv quyết định, không do pooling.

Ví dụ khác bài chính: input `(N,3,12,12)` qua Conv3→6 kernel3 padding0 thành `(N,6,10,10)`; MaxPool2 thành `(N,6,5,5)`; Flatten thành `(N,150)`.

Tự kiểm: kernel3 padding0 trên input 14 tạo chiều bao nhiêu? Nếu vô tình bỏ padding1 trong CNN bài chính, head Linear1568 có còn đúng không?

## 2. Parameter không phụ thuộc số ảnh

Conv2d có bias: `C_out * (C_in*K_h*K_w + 1)`.

Linear có bias: `D_out * (D_in + 1)`.

ReLU, pooling, Flatten và GAP không có weight học. Chúng vẫn ảnh hưởng thông tin và đường gradient. Parameter là một số đếm của model, không phải số activation của một batch.

Ví dụ Conv3→6 kernel3: `6*(3*3*3+1)=168`. Linear150→4: `4*(150+1)=604`. Với CNN bài chính, tự tính ba layer có weight rồi đối chiếu tổng **20.490**. Với GAP, đối chiếu **5.130**.

Tự kiểm: tăng batch 128→256 có làm tăng parameter không? Đổi kích thước ảnh có thể làm head Flatten tăng parameter như thế nào?

## 3. Flatten và GAP

Flatten giữ riêng từng vị trí trong feature map. Linear có thể gán trọng số khác nhau cho vùng trên/dưới của ảnh. GAP lấy trung bình mỗi kênh theo hai chiều không gian; head chỉ còn một số cho mỗi kênh.

GAP thường giảm kích thước head, nhưng trung bình có thể bỏ mất thông tin vị trí hữu ích. Với quần áo, một mẫu hiện ở vùng tay hoặc thân có thể có ý nghĩa khác nhau. Đây là giả thuyết để kiểm bằng validation; không xem là kết luận trước khi chạy.

Bài nhỏ: dùng feature map 2 kênh, mỗi kênh 2×2. Viết 8 số sau Flatten và 2 số sau GAP. Hoán vị vị trí trong mỗi kênh: GAP thay đổi không? Flatten thay đổi không?

## 4. Logits và softmax

Logits là điểm số thực của các lớp. Cross-entropy của PyTorch nhận logits; softmax dùng khi muốn hiển thị điểm từng lớp trong demo. Softmax không bổ sung khả năng học và không làm điểm thành xác suất đã hiệu chuẩn.

Ví dụ logits `[0,0,0]` cho ba điểm softmax bằng nhau. Cộng cùng một hằng số vào mọi logit không đổi softmax. Nếu đã softmax rồi đưa vào cross-entropy, phép tính không còn đúng với dự định của đề.

## 5. Quan sát ảnh và lỗi

Vẽ ảnh raw đúng grayscale; ảnh chuẩn hóa có thể chứa giá trị âm. Khi demo raw ảnh uint8 `(28,28)`, helper áp dụng cùng normalization của checkpoint. Không normalize hai lần.

Trong confusion matrix, hàng là nhãn thật, cột là dự đoán. Một ô Shirt→T-shirt/top lớn chỉ là điểm bắt đầu phân tích: đọc ảnh thật, xem kiểu dáng, rồi viết giả thuyết. Không suy ra model “không hiểu áo” chỉ từ một vài ví dụ.

Nguồn chính thức: [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html), [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html), [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
