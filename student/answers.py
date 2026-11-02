"""Điền TODO trong notebook hoặc module này. Không có lời giải dự phòng."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from shared.data import normalize
from shared.engine import unpack_batch


def build_vocab(texts, min_freq=2):
    # TODO N1: chỉ đếm token trong texts train; PAD=0, UNK=1.
    # Giữ tần suất >= min_freq; sort (-frequency, token), rồi gán ID từ 2.
    raise NotImplementedError('TODO N1: xây vocabulary từ train, không đọc validation/test.')


def encode_text(text, vocab, max_length=40):
    # TODO N2: normalize, split, truncate, UNK, pad; trả (list[int], length thật).
    # Input rỗng phải báo ValueError, không trả length=0 cho packed sequence.
    raise NotImplementedError('TODO N2: chuyển một câu thành ID và chiều dài thật.')


class FashionCNN(nn.Module):
    def __init__(self, variant):
        super().__init__()
        # TODO C1: hai khối Conv → ReLU → MaxPool; head theo variant flatten/gap.
        raise NotImplementedError('TODO C1: khai báo các layer CNN theo bảng shape trong đề.')

    def forward(self, x):
        # TODO C2: trả logits (N,10), không softmax trước cross-entropy.
        raise NotImplementedError('TODO C2: viết forward CNN.')


class SentimentLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO N3: Embedding với padding_idx=0, LSTM một chiều, Linear 3 lớp.
        raise NotImplementedError('TODO N3: khai báo các layer LSTM.')

    def forward(self, ids, lengths):
        # Khung packed sequence để không đọc hidden state tại PAD:
        # embedded = ...
        # packed = pack_padded_sequence(embedded, lengths.cpu(),
        #                              batch_first=True, enforce_sorted=False)
        # _, (h_n, _) = ...
        # return ...  # dùng h_n[-1]
        raise NotImplementedError('TODO N4: embedding → packed sequence → hidden cuối → logits.')


def build_model(config):
    return FashionCNN(config['variant']) if config['project'] == 'cnn' else SentimentLSTM(config)


def train_one_epoch(model, loader, optimizer, criterion, device, clip=None):
    # TODO T1: train mode; với từng batch dùng unpack_batch(batch, device).
    # zero_grad → logits → loss → backward → kiểm gradient → clip (nếu có) → step.
    # Trả loss trung bình theo số mẫu; không dùng validation để update weight.
    raise NotImplementedError('TODO T1: hoàn thiện một epoch train; xem Appendix A và D.')
