# Chạy trên Google Colab / Kaggle

Lấy toàn bộ repo hoặc ZIP học viên. Mở riêng ipynb từ GitHub/Drive không đồng nghĩa các module `shared/` đã có trong VM. Hoàn tất bootstrap dưới đây, sau đó chạy notebook từ đầu. Chưa có số đo runtime cloud trong bản bàn giao này; [Colab có tài nguyên thay đổi](https://research.google.com/colaboratory/faq.html).

## Google Colab bằng ZIP

1. Mở notebook CNN hoặc NLP bằng Colab (upload ipynb). Chọn GPU trong phần cài runtime nếu được cấp.
2. Upload `ai4b-mini-projects-student.zip` bằng bảng Files của runtime. ZIP chỉ chứa code, không chứa corpus.
3. Thêm ô bootstrap đầu notebook và chạy:

```python
from pathlib import Path
import zipfile, os
archive = Path('/content/ai4b-mini-projects-student.zip')
with zipfile.ZipFile(archive) as z:
    z.extractall('/content')
os.chdir('/content/ai4b-mini-projects-student')
```

4. Kiểm package trong đúng kernel:

```python
import importlib.util
missing = [p for p in ['torch','numpy','sklearn','pandas','pyarrow','matplotlib']
           if importlib.util.find_spec(p) is None]
print('Missing:', missing)
```

Nếu chỉ thiếu pyarrow, chạy `%pip install pyarrow`. Nếu thiếu package khác, cài đúng package (`sklearn` có tên cài `scikit-learn`). Không chạy nâng cấp tất cả hoặc cài lại torch khi runtime đã có bản hoạt động. Nếu môi trường local hoàn toàn mới, dùng requirements.txt.

5. Chạy từ ô môi trường. Notebook tự chọn cuda nếu có; lần đầu tải Fashion-MNIST/UIT-VSFC, lần sau dùng cache.
6. Notebook mặc định `RUN_DL=False`: baseline ML vẫn chạy; điền TODO rồi bật cờ để chạy DL. Test vẫn đóng.

Nếu anh/chị đã upload repo học viên lên GitHub, có thể clone repo công khai thay ZIP, rồi `os.chdir` tới root. Dùng URL thật của repo đã upload; không chèn token cá nhân vào notebook chia sẻ.

## Kaggle bằng ZIP code riêng

1. Import ipynb vào một Kaggle Notebook. Trong cài đặt notebook, bật GPU nếu có và Internet để helper tải dữ liệu.
2. Tạo input **riêng tư chỉ chứa ZIP code học viên**, gắn vào notebook. Không upload corpus UIT-VSFC thành dataset công khai.
3. Thêm ô bootstrap:

```python
from pathlib import Path
import zipfile, os
archives = list(Path('/kaggle/input').rglob('ai4b-mini-projects-student.zip'))
assert len(archives) == 1, 'Gắn đúng một ZIP code học viên vào Inputs.'
with zipfile.ZipFile(archives[0]) as z:
    z.extractall('/kaggle/working')
os.chdir('/kaggle/working/ai4b-mini-projects-student')
```

Kiểm/cài đúng package thiếu như Colab, rồi chạy notebook từ đầu. Không ghi output vào `/kaggle/input` (chỉ đọc). Output nằm trong repo dưới `/kaggle/working/.../runs/`. Lưu phiên bản notebook kèm output bằng chức năng của Kaggle và tải checkpoint từ Outputs sau khi kiểm vị trí. Xem [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

Nếu Internet không được cấp, chuẩn bị cache bằng helper trong phiên có Internet rồi chuyển cache qua kênh riêng được phép; không đổi nguồn/split để vượt lỗi mạng. Có thể đặt `AI4B_DATA_ROOT` tới cache trước ô môi trường.

## Lưu output và chạy tiếp

Mỗi epoch ghi `last.pt` (model, optimizer, RNG, config), và cập nhật `best.pt` nếu validation tốt hơn. Với NLP, checkpoint có vocabulary, preprocessing và labels. Sau khi restart runtime, giải nén/clone repo, khôi phục cùng output folder và config; `fit(..., resume=True)` tự dùng last.pt. Nếu đổi config, dùng thư mục mới, không sửa file để vượt kiểm fingerprint.

Tải output mà không kèm data:

```python
import shutil
shutil.make_archive('/content/ai4b-my-results', 'zip', 'runs/my_project')  # Colab
# Kaggle: đổi đích thành /kaggle/working/ai4b-my-results
```

Sau đó tải ZIP qua giao diện Files/Outputs. Tải cả notebook đã lưu output. Không commit các checkpoint vào repo phát bài dùng chung. Có thể đặt output ở Drive riêng trong Colab nếu đã mount bằng giao diện; không ghi đè bài bạn khác.

## Checklist giảng viên kiểm mỗi nền tảng

- [ ] Ghi ngày, GPU, Python và phiên bản thư viện.
- [ ] Chạy từ runtime sạch, tải/cache nguồn pin đúng revision và đủ số mẫu.
- [ ] Chạy hai baseline trước khi điền TODO.
- [ ] Chạy bản giảng viên đủ 10 epoch CNN và 12 epoch NLP, cả hai cấu hình seed 0.
- [ ] Lưu best/last, ngắt phiên, khôi phục cùng config và kiểm inference CPU.
- [ ] Kiểm PAD, UNK, input rỗng, ảnh sai shape; notebook học viên không tự chạy đáp án.
- [ ] Ghi thời gian đo và lưu notebook/checkpoint ra khỏi runtime.
- [ ] Ghi rõ đã kiểm nền tảng nào, bằng chứng ở đâu; chưa kiểm thì để “chưa kiểm”.
