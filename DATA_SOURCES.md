# Nguồn dữ liệu và tái hiện

## Fashion-MNIST

[Zalando Research Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), Han Xiao, Kashif Rasul, Roland Vollgraf (2017), *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, [arXiv:1708.07747](https://arxiv.org/abs/1708.07747). Repo nguồn có MIT license; xem license của nguồn khi sử dụng dữ liệu.

Helper tải bốn file IDX từ revision `b2617bb6d3ffa2e429640350f613e3291e10b141`, kiểm MD5 chuẩn và ghi SHA-256. 60.000 train gốc chia stratify thành 50.000 train / 10.000 validation với seed 42; 10.000 test giữ nguyên. Manifest lưu toàn bộ ID cùng revision và checksum.

## UIT-VSFC

[Dataset chính thức trên Hugging Face](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback), [trang tác giả](https://github.com/kietnv/uit-vsfc).

Nguyen, Kiet Van; Nguyen, Vu Duc; Nguyen, Phu X. V.; Truong, Tham T. H.; Nguyen, Ngan Luu-Thuy (2018). *UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis*. 10th International Conference on Knowledge and Systems Engineering (KSE), 19–24. DOI [10.1109/KSE.2018.8573337](https://doi.org/10.1109/KSE.2018.8573337).

Helper đọc bản chuyển parquet của chính dataset, revision `2d76906157232f87e9883fa37ebfc91ffdee1f83` (nhánh nguồn `refs/convert/parquet`), không chạy Python script từ xa. Giữ split 11.426 / 1.583 / 3.166 và mapping sentiment negative/neutral/positive. Cache giữ source.json; manifest lưu SHA-256 từng parquet và thống kê trùng nội dung NFC/lowercase.

Thẻ nguồn ghi license `unknown` tại thời điểm kiểm 04/11/2026. Repo bài tập và ZIP chỉ chứa code tải cùng trích dẫn; không phân phối lại corpus. Không tự tạo Kaggle Dataset công khai chứa cache khi chưa xác minh quyền phát lại. Việc tải được dữ liệu không tự quyết định quyền phát lại.

## Cache, test và checksum

`data/` không được đưa lên Git/ZIP phát hành. Lần tải đầu cần internet; lần sau helper dùng cache và xác minh checksum. Không tự sửa manifest khi hash không khớp. Một bộ cache thay đổi cần được điều tra trong thư mục mới.

Helper có thể tải file test để kiểm tính toàn vẹn benchmark; pipeline fit/validation không nhận test. Chỉ hàm đánh giá cuối có `open_test=True` sau khi xác minh lua_chon.json mới đo test. Thống kê trùng nội dung để audit không được dùng để sửa split hoặc tune model.

Digits của bài bổ trợ lấy qua [sklearn load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), chỉ đối chiếu số học NumPy/PyTorch, không dùng thay Fashion-MNIST.
