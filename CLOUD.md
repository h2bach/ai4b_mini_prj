# Chạy bài tập trên Google Colab / Kaggle

Mở notebook từ GitHub chỉ tải nội dung notebook. Bạn vẫn cần clone toàn bộ repo vào runtime để dùng các module `shared/`. Thêm ô bootstrap ở đầu notebook, chạy ô đó rồi mới chạy các ô bài tập có sẵn.

## Google Colab

1. Mở [notebook CNN](https://colab.research.google.com/github/h2bach/ai4b_mini_prj/blob/main/cnn/cnn_project.ipynb) hoặc [notebook LSTM](https://colab.research.google.com/github/h2bach/ai4b_mini_prj/blob/main/nlp/nlp_project.ipynb).
2. Lưu bản notebook cá nhân để giữ phần bạn chỉnh sửa. Chọn GPU trong phần cài đặt runtime nếu được cấp.
3. Thêm ô code đầu notebook:

```python
import os
import subprocess
from pathlib import Path

repo = Path('/content/ai4b_mini_prj')
if not (repo / '.git').exists():
    subprocess.run([
        'git', 'clone', 'https://github.com/h2bach/ai4b_mini_prj.git', str(repo)
    ], check=True)
os.chdir(repo)
print('Thư mục làm việc:', Path.cwd())
```

4. [Kiểm thư viện](#kiểm-thư-viện) rồi chạy các ô của bài từ đầu. Nếu không có GPU, notebook sẽ chọn CPU.
5. Chạy baseline trước; điền TODO rồi bật `RUN_DL=True`. Giữ `OPEN_TEST=False` tới khi đã phân tích validation và khóa lựa chọn.

Bạn cũng có thể upload bản ZIP tải từ GitHub qua Files, giải nén vào `/content`, rồi `os.chdir` tới thư mục có `shared/`, `cnn/` và `nlp/`. Không chỉ upload file ipynb.

## Kaggle

1. Import `cnn/cnn_project.ipynb` hoặc `nlp/nlp_project.ipynb` vào một Kaggle Notebook.
2. Bật Internet để clone repo và tải dữ liệu. Chọn GPU nếu có.
3. Thêm ô code đầu notebook:

```python
import os
import subprocess
from pathlib import Path

repo = Path('/kaggle/working/ai4b_mini_prj')
if not (repo / '.git').exists():
    subprocess.run([
        'git', 'clone', 'https://github.com/h2bach/ai4b_mini_prj.git', str(repo)
    ], check=True)
os.chdir(repo)
print('Thư mục làm việc:', Path.cwd())
```

4. Kiểm thư viện, rồi chạy notebook từ đầu. Ghi output trong `/kaggle/working`; `/kaggle/input` chỉ đọc.
5. Lưu notebook cùng output và lấy checkpoint qua Outputs sau khi chạy.

Nếu dùng ZIP code qua Inputs, giải nén ZIP sang `/kaggle/working` và chuyển tới thư mục repo đã giải nén. Không upload cache UIT-VSFC thành dataset công khai. Nếu runtime không có Internet, cần chuẩn bị cache qua kênh riêng được phép và giữ nguyên nguồn/split trong bài.

## Kiểm thư viện

Chạy trong đúng kernel đang dùng để làm bài:

```python
import importlib.util
missing = [p for p in ['torch', 'numpy', 'sklearn', 'pandas', 'pyarrow', 'matplotlib']
           if importlib.util.find_spec(p) is None]
print('Thư viện còn thiếu:', missing)
```

Chỉ cài package thiếu. Ví dụ nếu thiếu pyarrow:

```python
%pip install pyarrow
```

Tên cài của `sklearn` là `scikit-learn`. Không nâng cấp hàng loạt môi trường cloud hoặc cài lại torch khi bản sẵn có đã chạy được. Môi trường local mới có thể dùng `requirements.txt` theo hướng dẫn trong [README](README.md).

## Lưu output và chạy tiếp

Notebook mặc định ghi `runs/my_project/` bên trong repo. Các file cần giữ gồm notebook bạn đã sửa, `best.pt`, `last.pt`, history, metric, `lua_chon.json`, cùng vocabulary NLP. Checkpoint lưu preprocessing và label mapping.

Sau mỗi phiên, lưu notebook cá nhân và tải output khỏi máy runtime. Tạo ZIP kết quả:

```python
from pathlib import Path
import shutil

# Colab dùng /content; Kaggle dùng /kaggle/working.
destination = Path('/content') if Path('/content').exists() else Path('/kaggle/working')
archive = shutil.make_archive(str(destination / 'ai4b-my-results'), 'zip', 'runs/my_project')
print('Tải file này qua Files/Outputs:', archive)
```

Để chạy tiếp sau khi ngắt phiên: clone/giải nén repo, khôi phục output về cùng thư mục, nạp lại phần code TODO bạn đã hoàn thiện và giữ nguyên cấu hình. `fit(..., resume=True)` dùng `last.pt` để khôi phục model, optimizer và RNG. Đổi cấu hình phải dùng output folder mới.

Nếu muốn lưu ở đường dẫn khác, đặt trước ô môi trường của notebook:

```python
import os
os.environ['AI4B_DATA_ROOT'] = '/duong/dan/cache'  # Thư mục chứa cnn/ và nlp/.
os.environ['AI4B_OUTPUT_ROOT'] = '/duong/dan/ket-qua'
```

Thay bằng đường dẫn thực tế, có quyền ghi. Kết quả benchmark phải dùng đủ split và budget theo đề; `AI4B_SMOKE=1` chỉ là chế độ kiểm kỹ thuật subset.

## Tự kiểm trước khi kết thúc phiên

- [ ] Baseline chạy được; đã đọc metric validation theo từng lớp.
- [ ] Notebook cá nhân đã lưu toàn bộ phần TODO bạn hoàn thiện.
- [ ] Checkpoint và kết quả đã được tải/lưu ra ngoài runtime.
- [ ] Biết vị trí output và cấu hình để khôi phục phiên sau.
- [ ] Có thể reload checkpoint bằng CPU để predict.
- [ ] Test chỉ mở sau khi khóa lựa chọn; phần còn thiếu được đối chiếu với checklist bài nộp.

Tài nguyên cloud có thể thay đổi; không lấy thời gian máy khác làm thời gian thực đo của bạn. Nguồn sử dụng nền tảng: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).
