import torch
import numpy as np
import os
from torch.utils.data import Dataset

class Global:
    """
    Holds global mean and standard deviation parameters for training and denormalization.
    """
    train_mean = 0.0
    train_std = 1.0

# --- Normalization Functions ---

def train_offset_normalization(data):
    """
    Normalizes spatial coordinates based on training data.
    """
    mean = data[:, :, 1:].mean(axis=(0, 1))
    data[:, :, 1:] -= mean
    std = data[:, :, 1:].std(axis=(0, 1))
    data[:, :, 1:] /= std
    return mean, std, data

def valid_offset_normalization(mean, std, data):
    """
    Normalizes validation dataset based on the training mean and standard deviation.
    """
    data[:, :, 1:] -= mean
    data[:, :, 1:] /= std
    return data

def data_denormalization(mean, std, data):
    """
    Denormalizes offsets back into coordinates space for visualization.
    """
    data[:, :, 1:] *= std
    data[:, :, 1:] += mean
    return data

# --- Dataset Class ---

class HandwritingDataset(Dataset):
    """
    Custom PyTorch Dataset class designed to process handwriting strokes and sentences
    for training the handwriting generation model.
    """
    def __init__(self, data_path, split='train', text_req=False, debug=False, max_seq_len=300, data_aug=False):
        self.text_req = text_req
        self.max_seq_len = max_seq_len
        self.data_aug = data_aug

        # Load strokes array and texts corpus
        strokes = np.load(os.path.join(data_path, 'strokes.npy'), allow_pickle=True, encoding='bytes')
        with open(os.path.join(data_path, 'sentences.txt')) as file:
            texts = file.read().splitlines()

        lengths = [len(stroke) for stroke in strokes]
        max_len = np.max(lengths)
        n_total = len(strokes)

        mask_shape = (n_total, max_len)
        mask = np.zeros(mask_shape, dtype=np.float32)

        char_seqs = [list(char_seq) for char_seq in texts]
        char_seqs = np.array(char_seqs, dtype=object)

        char_lens = [len(char_seq) for char_seq in char_seqs]
        max_char_len = np.max(char_lens)

        mask_shape = (n_total, max_char_len)
        char_mask = np.zeros(mask_shape, dtype=np.float32)

        inp_text = np.ndarray((n_total, max_char_len), dtype='<U1')
        inp_text[:, :] = ' '

        data_shape = (n_total, max_len, 3)
        data = np.zeros(data_shape, dtype=np.float32)

        for i, (seq_len, text_len) in enumerate(zip(lengths, char_lens)):
            mask[i, :seq_len] = 1.
            data[i, :seq_len] = strokes[i]
            char_mask[i, :text_len] = 1.
            inp_text[i, :text_len] = char_seqs[i]

        self.id_to_char, self.char_to_id = self.build_vocab(inp_text)
        self.vocab_size = len(self.id_to_char)

        # Shuffle indexes
        idx_permute = np.random.permutation(n_total)
        data = data[idx_permute]
        mask = mask[idx_permute]
        inp_text = inp_text[idx_permute]
        char_mask = char_mask[idx_permute]

        if debug:
            data = data[:64]
            mask = mask[:64]
            inp_text = inp_text[:64]
            char_mask = char_mask[:64]

        n_train = int(0.9 * data.shape[0])
        self._data = data
        if split == 'train':
            self.dataset = data[:n_train]
            self.mask = mask[:n_train]
            self.texts = inp_text[:n_train]
            self.char_mask = char_mask[:n_train]
            Global.train_mean, Global.train_std, self.dataset = train_offset_normalization(
                self.dataset)
        elif split == 'valid':
            self.dataset = data[n_train:]
            self.mask = mask[n_train:]
            self.texts = inp_text[n_train:]
            self.char_mask = char_mask[n_train:]
            self.dataset = valid_offset_normalization(
                Global.train_mean, Global.train_std, self.dataset)

    def __len__(self):
        return self.dataset.shape[0]

    def idx_to_char(self, id_seq):
        return np.array([self.id_to_char[id] for id in id_seq])

    def char_to_idx(self, char_seq):
        return np.array([self.char_to_id[char] for char in char_seq]).astype(np.float32)

    def build_vocab(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(text)
        unique_char = sorted(counter)
        vocab_size = len(unique_char)

        id_to_char = dict(zip(np.arange(vocab_size), unique_char))
        char_to_id = dict([(v, k) for (k, v) in id_to_char.items()])
        return id_to_char, char_to_id
