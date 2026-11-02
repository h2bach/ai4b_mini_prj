# AI4B Mini Projects — Hướng dẫn dành cho học viên

Repo này là bộ bài tập thực hành cuối khóa AI4B. Mỗi học viên tự hoàn thành **hai mini projects trong 14 ngày**: CNN phân loại quần áo và LSTM phân loại góp ý tiếng Việt.

Bạn được cung cấp **đề bài, baseline ML chạy hoàn chỉnh và notebook DL có TODO**. Khi làm bài, bạn tự hoàn thiện phần CNN/LSTM, chạy thí nghiệm, phân tích lỗi và giải thích lựa chọn. Kết quả DL không bắt buộc phải cao hơn baseline ML.

## 1. Bắt đầu từ đâu?

| Thứ tự | Bạn cần làm | Tài liệu |
|---|---|---|
| 1 | Lấy toàn bộ repo và chuẩn bị môi trường | [Cách mở repo](#2-lấy-repo-và-chuẩn-bị-môi-trường), [Kaggle/Colab](CLOUD.md) |
| 2 | Ôn tensor, autograd và một bước cập nhật weight | [NumPy → PyTorch](lessons/01_numpy_to_pytorch.ipynb) — khoảng 45 phút |
| 3 | Đọc đề CNN, mở notebook, chạy baseline rồi điền TODO | [Đề CNN](cnn/DE_BAI.md), [notebook CNN](cnn/cnn_project.ipynb) |
| 4 | Học token, embedding, hidden state và padding | [Bổ trợ NLP](lessons/02_nlp_foundations.ipynb) — khoảng 60 phút |
| 5 | Đọc đề LSTM, chạy baseline văn bản rồi hoàn thiện model | [Đề LSTM](nlp/DE_BAI.md), [notebook LSTM](nlp/nlp_project.ipynb) |
| 6 | Khóa lựa chọn, đánh giá cuối, viết báo cáo và nộp bài | [Mẫu báo cáo](student/REPORT_TEMPLATE.md), [checklist bài nộp](student/SUBMISSION_CHECKLIST.md) |

**Nếu chưa biết PyTorch, hãy học bài bổ trợ trước. Nếu gặp lỗi khi chạy, tra [phụ lục sửa lỗi](appendix/E_TROUBLESHOOTING.md).**

## 2. Lấy repo và chuẩn bị môi trường

### Clone bằng Git

```bash
git clone https://github.com/h2bach/ai4b_mini_prj.git
cd ai4b_mini_prj
```

Bạn cũng có thể chọn **Code → Download ZIP** trên GitHub, giải nén rồi mở thư mục chứa `README.md`, `cnn/`, `nlp/` và `shared/`. Đừng chỉ tải một file notebook: notebook cần các module đi kèm để tải dữ liệu, chạy baseline và ghi kết quả.

### Dùng Google Colab hoặc Kaggle

Làm theo [CLOUD.md](CLOUD.md) để clone repo vào runtime, chọn thư mục làm việc và kiểm tra thư viện. Sau khi mở notebook trên cloud, **thêm ô bootstrap theo hướng dẫn trước khi chạy các ô có sẵn**.

| Project | Notebook trong repo | Mở bằng Colab |
|---|---|---|
| CNN | [cnn/cnn_project.ipynb](cnn/cnn_project.ipynb) | [Mở CNN trên Colab](https://colab.research.google.com/github/h2bach/ai4b_mini_prj/blob/main/cnn/cnn_project.ipynb) |
| LSTM | [nlp/nlp_project.ipynb](nlp/nlp_project.ipynb) | [Mở LSTM trên Colab](https://colab.research.google.com/github/h2bach/ai4b_mini_prj/blob/main/nlp/nlp_project.ipynb) |

### Dùng VS Code trên máy cá nhân

Cần Python từ 3.10 trở lên. Tại thư mục gốc của repo:

```bash
python -m venv .venv
```

Kích hoạt môi trường theo hệ điều hành:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Sau đó cài thư viện và đăng ký kernel:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ai4b-student --display-name "AI4B Student"
```

Mở cả thư mục repo trong VS Code, mở notebook và chọn kernel **AI4B Student**. Nếu dùng Jupyter, chọn Python của môi trường này. PyTorch dùng GPU cần bản phù hợp thiết bị; CPU vẫn chạy được. Trên cloud đã có torch, chỉ cài package còn thiếu theo CLOUD.md.

## 3. Các thành phần của repo

```text
ai4b_mini_prj/
├── README.md                 # Bắt đầu tại đây
├── CLOUD.md                  # Chạy bằng Colab/Kaggle, lưu và khôi phục output
├── DATA_SOURCES.md           # Nguồn dữ liệu, trích dẫn, manifest và checksum
├── requirements.txt          # Thư viện để làm bài
├── cnn/
│   ├── DE_BAI.md             # Đề bài, yêu cầu và rubric CNN
│   ├── cnn_project.ipynb     # Notebook CNN: baseline hoàn chỉnh + TODO
│   └── config.json           # Cấu hình thí nghiệm CNN
├── nlp/
│   ├── DE_BAI.md             # Đề bài, yêu cầu và rubric LSTM
│   ├── nlp_project.ipynb     # Notebook NLP: baseline hoàn chỉnh + TODO
│   └── config.json           # Cấu hình thí nghiệm LSTM
├── student/
│   ├── answers.py            # Bản mẫu các hàm có TODO
│   ├── REPORT_TEMPLATE.md    # Mẫu báo cáo tối đa 2 trang/project
│   └── SUBMISSION_CHECKLIST.md
├── shared/
│   ├── data.py               # Tải dữ liệu, đọc split và chuẩn hóa
│   ├── baselines.py          # Majority và logistic regression / TF-IDF
│   ├── engine.py             # Điều phối train, validation, checkpoint, inference
│   ├── metrics.py            # Metric, biểu đồ và phân tích mẫu sai
│   ├── protocol.py           # Đăng ký quy tắc, khóa lựa chọn và mở test
│   ├── kiem_tra.py            # Kiểm shape, parameter, padding, UNK, input rỗng
│   ├── utils.py              # Tiện ích cấu hình và ghi/đọc kết quả
│   └── manifests/            # Split cố định và dấu vết nguồn dữ liệu
├── scripts/
│   └── run_baselines.py      # Chạy baseline độc lập từ terminal
├── lessons/                  # Hai notebook bổ trợ và ví dụ NumPy/PyTorch
└── appendix/                 # Phụ lục A–F để tra cứu khi làm bài
```

Bạn chủ yếu làm việc trong **hai notebook project** và viết báo cáo. `shared/` cung cấp hạ tầng dùng chung; hãy đọc helper khi cần hiểu pipeline. Giữ split, preprocessing và ngân sách thí nghiệm theo đề để các kết quả so sánh được công bằng.

`student/answers.py` là **template TODO**, không phải file đáp án đã giải. Notebook có bản sao template trong ô code. Cách bắt đầu đơn giản nhất là sửa trực tiếp ô TODO trong notebook; sửa module `answers.py` sẽ không tự cập nhật ô notebook. Nếu chọn viết trong module, cần chủ động import lại các hàm sau mỗi lần sửa.

## 4. Hai bài tập và baseline được cung cấp

| Nội dung | CNN — Fashion-MNIST | LSTM — UIT-VSFC |
|---|---|---|
| Đầu vào | Ảnh xám 28×28 | Câu góp ý tiếng Việt |
| Đầu ra | 10 lớp quần áo | negative / neutral / positive |
| Baseline 1 | Dự đoán lớp nhiều nhất trong train | Dự đoán lớp nhiều nhất trong train |
| Baseline 2 | Logistic regression trên pixel | TF-IDF + logistic regression |
| Bạn tự làm | Kiến trúc CNN, forward và bước train | Vocabulary, mã hóa chuỗi, LSTM, forward và bước train |
| Thí nghiệm bắt buộc | Flatten so với GAP | Cross-entropy thường so với có trọng số |
| Seed bắt buộc | 0 cho cả hai cấu hình | 0 cho cả hai cấu hình |

### Chạy riêng baseline trước

Từ thư mục gốc repo, dùng Python của môi trường đã cài thư viện:

```bash
python -m scripts.run_baselines --project cnn
python -m scripts.run_baselines --project nlp
```

Lần đầu cần Internet để tải dữ liệu. Helper tự dùng cache ở những lần sau. Baseline không phụ thuộc việc bạn đã hoàn thiện TODO hay chưa; CLI cũng chạy cùng baseline với notebook.

Kết quả validation được ghi vào `runs/my_project/<project>/baselines/`, gồm `baselines.json` và CSV dự đoán theo sample ID. Các file `.joblib` lưu model ML đã fit. Với cùng dữ liệu và output có sẵn, chương trình dùng lại baseline đã lưu.

### Tiếp tục phần DL trong notebook

1. Chạy các ô từ đầu: môi trường → dữ liệu → baseline.
2. Đọc bảng shape và tính parameter theo đề trước khi điền TODO.
3. Hoàn thiện model, forward và train step. NLP cần hoàn thiện thêm vocabulary và mã hóa chuỗi.
4. Đặt `RUN_DL=True`, rồi chạy lại từ ô định nghĩa các hàm trở đi. Khi cần kiểm trọn pipeline, restart kernel và chạy từ đầu.
5. Chạy đủ hai cấu hình seed 0, đọc learning curve và phân tích ít nhất 5 mẫu sai trên validation.
6. Sau khi phân tích xong, đặt `LOCK_SELECTION=True` để ghi `lua_chon.json`. Chỉ sau đó mới đặt `OPEN_TEST=True` cho lượt đánh giá cuối.

Khi `RUN_DL=False`, phần baseline vẫn chạy, còn phần DL in hướng dẫn hoàn thiện TODO. Khi bật cờ nhưng chưa điền đủ, `NotImplementedError` chỉ rõ phần còn thiếu.

## 5. Dữ liệu và output nằm ở đâu?

Hai thư mục này được tạo khi chạy và đã có trong `.gitignore`:

| Thư mục/file | Dùng để làm gì? |
|---|---|
| `data/cnn/`, `data/nlp/` | Cache dữ liệu tải về và manifest đã kiểm |
| `runs/my_project/<project>/baselines/` | Metric validation, dự đoán và model ML |
| `runs/my_project/<project>/<variant>/seed_0/` | Output mỗi cấu hình DL |
| `best.pt` | Checkpoint được chọn theo macro-F1 validation |
| `last.pt` | Checkpoint cuối epoch để tiếp tục cùng cấu hình |
| `history.json`, `summary.json`, `validation.csv` | Diễn biến train, kết quả validation và dự đoán |
| `lua_chon.json` | Lựa chọn được khóa trước khi đánh giá test |
| `test_report.json` | Kết quả cuối sau khi mở test |

Giữ checkpoint, cấu hình, preprocessing, label mapping và vocabulary NLP để tái hiện. Lưu notebook có output và tải kết quả khỏi runtime cloud trước khi kết thúc phiên; xem [cách lưu/khôi phục](CLOUD.md#lưu-output-và-chạy-tiếp).

## 6. Phụ lục để học và tra cứu

| Bạn đang cần… | Đọc phần này |
|---|---|
| Hiểu tensor, autograd, DataLoader và optimizer | [A — PyTorch](appendix/A_PYTORCH.md) |
| Tính shape, parameter và hiểu Flatten/GAP | [B — CNN](appendix/B_CNN_SHAPES.md) |
| Hiểu token, vocabulary, embedding, LSTM và PAD | [C — NLP](appendix/C_NLP.md) |
| Hiểu F1, leakage, seed và chọn checkpoint | [D — Đánh giá](appendix/D_EVALUATION.md) |
| Sửa lỗi môi trường, shape, GPU và checkpoint | [E — Sửa lỗi](appendix/E_TROUBLESHOOTING.md) |
| Tìm nguồn học thêm và thuật ngữ | [F — Tài liệu](appendix/F_RESOURCES.md) |

## 7. Lịch làm bài và sản phẩm nộp

| Ngày | Công việc |
|---|---|
| 1–2 | Chuẩn bị môi trường, học cầu nối PyTorch, chạy baseline CNN |
| 3–4 | Hoàn thiện CNN, kiểm shape/parameter và train |
| 5–6 | So sánh GAP, phân tích ảnh sai và kiểm tiến độ lần 1 |
| 7–8 | Học bổ trợ NLP, chạy baseline TF-IDF |
| 9–10 | Hoàn thiện biểu diễn chuỗi và LSTM |
| 11–12 | Thử class weighting, phân tích lỗi, khóa lựa chọn; kiểm tiến độ lần 2 |
| 13 | Đánh giá test sau khóa, kiểm tái hiện và viết báo cáo |
| 14 | Nộp hai bài và demo tổng cộng tối đa 5 phút |

Mỗi project nộp notebook đã hoàn thiện có output, checkpoint và preprocessing cần thiết, vocabulary NLP, `lua_chon.json`, bảng kết quả và báo cáo tối đa **2 trang**. Dùng [mẫu báo cáo](student/REPORT_TEMPLATE.md) và [checklist](student/SUBMISSION_CHECKLIST.md) để rà soát. Chi tiết yêu cầu và rubric nằm trong từng đề.

Mean/std, TF-IDF, vocabulary và class weights chỉ học từ **train**. Chọn model trên **validation**; dùng **test** cho báo cáo cuối sau khi khóa lựa chọn. Báo accuracy kèm số đúng/tổng, macro-F1 và metric từng lớp; giải thích bằng số đo của chính bạn. Xem [nguồn dữ liệu và trích dẫn](DATA_SOURCES.md).
