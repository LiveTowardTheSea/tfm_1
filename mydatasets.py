import spacy
import torchtext.data as data
from torchtext.datasets import Multi30k
import torch
import config
import numpy as np
import os
import torchtext.vocab as vocab
import torch.nn as nn

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


class mydatasets(data.Dataset):
    def __init__(self, examples=None, fields=None):
        if examples is None:
            examples = Multi30k(path="../.data/multi30k/train", exts=('.de', '.en'), fields=fields).examples
        super(mydatasets, self).__init__(examples, fields=fields)

    @classmethod
    def splits(cls, config, fields):
        entire_examples = cls(fields=fields).examples
        if config.shuffle:
            np.random.shuffle(entire_examples)
        test_idx = int(len(entire_examples) * config.test_ratio // 1)
        dev_idx = int(len(entire_examples) * config.dev_ratio // 1) + test_idx
        #print(test_idx, dev_idx)
        return (cls(entire_examples[:-dev_idx], fields=fields), cls(entire_examples[-dev_idx:-test_idx], fields=fields),
                cls(entire_examples[-test_idx:], fields=fields))


def get_iter(config, device):
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = data.Field(init_token="<sos>", eos_token="<eos>",
                     lower=True, tokenize=tokenize_de)
    TRG = data.Field(init_token="<sos>", eos_token="<eos>",
                     lower=True, tokenize=tokenize_en)
    fields = [('src', SRC), ('trg', TRG)]
    train_data, dev_data, test_data = mydatasets.splits(config, fields)
    print("train—size:{}, "
          "dev_size:{}, "
          "test_size:{} ".format(len(train_data), len(dev_data), len(test_data)))  # 26100 1450 1450
    SRC.build_vocab(train_data)
    TRG.build_vocab(train_data)
    print("German vocab_size:", len(SRC.vocab))
    print("English vocab_size:", len(TRG.vocab))
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_data, dev_data, test_data),
                                                                 batch_sizes=(config.batch_size,config.batch_size,config.batch_size),
                                                                 device=device,
                                                                 sort_key=lambda x: len(x.src))
    pad_index = SRC.vocab.stoi[SRC.pad_token]
    sos_token = SRC.vocab.stoi[SRC.init_token]
    return train_iter, dev_iter, test_iter, len(SRC.vocab), len(TRG.vocab), pad_index, sos_token


if __name__ == "__main__":
    config = config.Config()
    device = torch.device('cuda')
    l1, l2, l3, src_len, trg_len, pad, sos = get_iter(config, device)
    src_max_len = 0
    trg_max_len = 0
    for data2 in l3:
        if data2.src.shape[0] > src_max_len:
            src_max_len = data2.src.shape[0]
        if data2.trg.shape[0] > trg_max_len:
            trg_max_len = data2.trg.shape[0]

    print(src_max_len, trg_max_len)