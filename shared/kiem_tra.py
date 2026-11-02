"""Kiểm các hợp đồng công khai; không thay học viên viết model/train."""
import numpy as np
import torch
from shared.engine import predict_image, predict_text


def check_model(model, xs, n_classes, expected_params):
    model.eval()
    with torch.no_grad():
        logits = model(*xs)
    assert logits.shape == (len(xs[0]), n_classes), 'Sai shape logits'
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all(), 'Sai dtype/giá trị logits'
    assert sum(p.numel() for p in model.parameters()) == expected_params, 'Sai số parameter'
    print('Đạt kiểm shape, dtype, parameter và logits hữu hạn.')


def check_padding(model, encode, vocab):
    text = 'giảng viên nhiệt tình'
    a, length = encode(text, vocab, 40)
    b, length_b = encode(text, vocab, 70)
    c, length_c = encode('môn học cần thêm ví dụ và bài tập thực hành', vocab, 70)
    model = model.cpu().eval()
    with torch.no_grad():
        p = model(torch.tensor([a]), torch.tensor([length]))
        q = model(torch.tensor([b, c]), torch.tensor([length_b, length_c]))[:1]
    assert torch.allclose(p, q, atol=1e-6, rtol=1e-5), 'Hidden state phụ thuộc PAD hoặc thứ tự batch'
    try:
        encode('   ', vocab, 40)
    except ValueError:
        pass
    else:
        raise AssertionError('Cần từ chối input rỗng bằng ValueError')
    ids, n = encode('zzzz_ai4b_unknown', vocab, 40)
    assert n == 1 and ids[0] == 1, 'Chưa xử lý UNK đúng'
    print('Đạt kiểm padding, input rỗng và UNK.')
