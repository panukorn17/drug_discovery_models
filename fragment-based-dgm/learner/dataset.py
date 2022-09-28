import time
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from utils.filesystem import load_dataset
from .skipgram import Vocab


class DataCollator:
    def __init__(self, vocab):
        self.vocab = vocab

    def merge(self, sequences):
        sequences = sorted(sequences, key=len, reverse=True)
        lengths = [len(seq) for seq in sequences]
        padded_seqs = np.full((len(sequences), max(lengths)), self.vocab.PAD)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        #print(padded_seqs)
        return torch.LongTensor(padded_seqs), lengths

    def __call__(self, data):
        # seperate source and target sequences
        src_seqs, tgt_seqs, data_sample, tgt_str = zip(*data)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        src_seqs, src_lengths = self.merge(src_seqs)
        tgt_seqs, tgt_lengths = self.merge(tgt_seqs)
        #return src_seqs, tgt_seqs, src_lengths
        ### Teddy Code
        return src_seqs, tgt_seqs, src_lengths, data_sample, tgt_str
        ###


class FragmentDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, config):
        """Reads source and target sequences from csv files."""
        self.config = config

        data = load_dataset(config, kind='train')
        # data = data[data.n_fragments>3]
        self.data = data.reset_index(drop=True)
        self.size = self.data.shape[0]

        self.vocab = None

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        seq = self.data.fragments[index].split(" ")
        #print(seq)
        ### Teddy code
        #print(self.data.fragments[index])
        ###
        seq = self.vocab.append_delimiters(seq)
        #print(seq)
        #print(seq[:-1])
        #print(seq[1:])
        src = self.vocab.translate(seq[:-1])
        #print("source: ", self.vocab.translate(src))
        #print(src)
        tgt = self.vocab.translate(seq[1:])
        #print("target: ", self.vocab.translate(tgt))
        #print(tgt)
        #return src, tgt
        ### Teddy code
        return src, tgt, index, seq[1:-1]

    def __len__(self):
        return self.size

    def get_loader(self):
        start = time.time()
        collator = DataCollator(self.vocab)
        loader = DataLoader(dataset=self,
                            collate_fn=collator,
                            batch_size=self.config.get('batch_size'),
                            num_workers=24,
                            shuffle=True,
                            pin_memory=False)
        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Data loaded. Size: {self.size}. '
              f'Time elapsed: {elapsed}.')
        return loader

    def get_vocab(self):
        start = time.time()
        if self.vocab is None:
            try:
                self.vocab = Vocab.load(self.config)
            except Exception:
                self.vocab = Vocab(self.config, self.data)

        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Vocab created/loaded. '
              f'Size: {self.vocab.get_size()}. '
              f'Effective size: {self.vocab.get_effective_size()}. '
              f'Time elapsed: {elapsed}.')

        return self.vocab
