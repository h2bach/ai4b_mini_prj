# Appendix A — NumPy → PyTorch

## 1. Tensor và dtype

Tensor có shape và dtype giống array NumPy, thêm device và khả năng ghi đồ thị đạo hàm. Ảnh và weight thường float32; ID token và label phải int64. `float32` không có nghĩa là ảnh đã chia 255: dtype và normalization là hai quyết định riêng.

```python
import torch
pixels = torch.tensor([[0., 255.]], dtype=torch.float32) / 255
labels = torch.tensor([0], dtype=torch.long)
print(pixels.shape, labels.dtype, pixels.device)
```

Tự kiểm: input `(128,1,28,28)` có bao nhiêu số? Vì sao không đổi ID token thành float rồi đưa vào Embedding?

## 2. Autograd và gradient cộng dồn

```python
w = torch.tensor([2.], requires_grad=True)
loss = (3*w - 1).square().mean()
loss.backward()
print(w.grad)  # 30: 2*(3*2-1)*3
```

PyTorch ghi phép toán trên tensor cần gradient và áp dụng chain rule khi backward. `.grad` tích lũy; nếu backward ở một forward mới mà không xóa gradient, giá trị sẽ cộng thêm. `detach()` tạo tensor tách khỏi đồ thị; dùng `.item()` để ghi một số scalar trong log.

Bài tự kiểm: đổi w=1, tính tay gradient trước khi chạy. Nếu loss dùng `model(x).detach()`, đường gradient về model còn không?

## 3. Module, optimizer và chế độ chạy

`nn.Module` quản lý parameter. Khai báo layer thuộc `self` để PyTorch nhận ra weight. Tạo model trước, optimizer sau và truyền `model.parameters()`. Không tạo lại layer trong forward.

Một batch train cần đi qua: xóa gradient → forward → loss → backward → kiểm gradient → clipping nếu yêu cầu → optimizer update. Bài chính yêu cầu bạn tự chuyển quy trình này thành code.

`model.train()` và `model.eval()` điều khiển hành vi layer như Dropout/BatchNorm. `torch.no_grad()` ngừng ghi đồ thị. Hai việc có mục đích khác nhau; validation thường cần cả eval và no_grad. Adam có trạng thái riêng, vì vậy resume cần cả optimizer state.

Tự kiểm: loss giảm nhưng weight model dùng để predict không đổi có thể do đâu? Kiểm optimizer đang giữ parameter của instance nào.

## 4. Dataset và DataLoader

Dataset ánh xạ index → một mẫu. DataLoader ghép thành batch, có thể shuffle train. Validation/test giữ thứ tự cố định để nối prediction với sample ID.

```python
from torch.utils.data import DataLoader, TensorDataset
x = torch.arange(12, dtype=torch.float32).reshape(4,3)
y = torch.tensor([0,1,0,1])
loader = DataLoader(TensorDataset(x,y), batch_size=2, shuffle=False)
for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.tolist())
```

Trong bộ bài tập, ảnh batch trả `(images, labels)`, NLP trả `(ids, lengths, labels)`. `unpack_batch` chuyển device; lengths vẫn cần `.cpu()` trước packed sequence. Không tráo label hoặc sample ID khi shuffle.

## 5. Bài đối chiếu Digits

Mở [notebook cầu nối](../lessons/01_numpy_to_pytorch.ipynb). Ví dụ có Conv1→8, ReLU, AvgPool2, Flatten128, Linear10. Dùng cùng batch và weight để so forward, CE, gradient từng parameter, gradient input và một bước SGD bằng NumPy/PyTorch. Mục tiêu sai số CPU <1e-6; ví dụ chạy float64 để giảm nhiễu số học.

Đây là kiểm implementation, không phải benchmark chất lượng phân loại Digits. Sau khi hiểu, bạn tự viết train step cho CNN/LSTM dùng Adam như đề bài.

Đọc thêm: [Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html), [Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
