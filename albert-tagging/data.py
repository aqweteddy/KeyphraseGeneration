import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
from sklearn.utils import shuffle


class KpBioDataset(Dataset):
    BIO_TO_IDX = {'B': 1, 'I':2, 'O': 0}
    IDX_TO_BIO = {1: 'B', 2: 'I', 'O': 0}

    def __init__(self, file, tokenizer, mode: str = 'train', maxlen=512, encoded=False):
        if encoded:
            return
        self.maxlen = maxlen
        self.mode = mode

        self.input_ids = []
        self.seg_ids = []
        self.masks = []
        self.bio = []

        self.tokenizer = tokenizer
        with open(file) as f:
            for idx, line in enumerate(tqdm(f.readlines())):
                line = line.strip()
                if mode == 'train':
                    if idx % 2 == 0:
                        inp_ids, type_ids, attention_mask = self.encode(line)
                        self.input_ids.append(inp_ids)
                        self.seg_ids.append(type_ids)
                        self.masks.append(attention_mask)
                    else:
                        self.bio.append(self.encode_bio(line))
                else:
                    inp_ids, type_ids, attention_mask = self.encode(line)
                    self.input_ids.append(inp_ids)
                    self.seg_ids.append(type_ids)
                    self.masks.append(attention_mask)

    @staticmethod
    def from_encoded(file: str, tokenizer, mode: str = 'train'):
        ds = KpBioDataset('q', tokenizer, encoded=True)
        print('reading file...')
        data = pd.read_json(file)
        data = shuffle(data)
        print('finished')
        ds.tokenizer = tokenizer
        ds.mode = mode
        ds.input_ids = data['input_ids'].to_list()
        ds.seg_ids = data['seg_ids'].to_list()
        ds.masks = data['masks'].to_list()
        ds.maxlen = len(ds.input_ids[0])

        if mode == 'train':
            ds.bio = data['bio'].to_list()
        
        assert len(ds.input_ids) == len(ds.seg_ids)
        assert len(ds.input_ids) == len(ds.masks)
        assert len(ds.input_ids) == len(ds.bio)
        return ds

    def encode_bio(self, bio):
        bio = [self.BIO_TO_IDX[char] for char in bio]
        bio = [0] + bio # [cls]
        if len(bio) < self.maxlen:
            padding_length = self.maxlen - len(bio)
            bio += [0] * padding_length
        
        bio = bio[:self.maxlen]
        bio[-1] = 0 # [sep]
        return bio

    def __getitem__(self, idx: int):
        if self.mode == 'train':
            return torch.tensor(self.input_ids[idx]), torch.tensor(self.seg_ids[idx]), torch.tensor(self.masks[idx]), torch.tensor(self.bio[idx])
        else:
            return torch.tensor(self.input_ids[idx]), torch.tensor(self.seg_ids[idx]), torch.tensor(self.masks[idx])

    def __len__(self):
        return len(self.input_ids)

    def save(self, file: str):
        if self.mode == 'train':
            df = pd.DataFrame({'input_ids': self.input_ids,
                        'seg_ids': self.seg_ids, 'masks': self.masks, 'bio': self.bio})
        else:
            df = pd.DataFrame({'input_ids': self.input_ids, 'seg_ids': self.seg_ids,
                        'masks': self.masks})
        df.to_json(file)

    def encode(self, text: str):
        inp = tokenizer.encode_plus(
            text=text, add_special_tokens=True, max_length=self.maxlen)
        inp_ids, type_ids = inp['input_ids'], inp['token_type_ids']
        attention_mask = inp['attention_mask']

        padding_length = self.maxlen - len(inp_ids)
        inp_ids = inp_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        type_ids = type_ids + ([0] * padding_length)

        assert len(inp_ids) == self.maxlen
        assert len(type_ids) == self.maxlen
        assert len(attention_mask) == self.maxlen
        return inp_ids, type_ids, attention_mask


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('./albert_base')
    ds = KpBioDataset('./data/test.txt', tokenizer, maxlen=512)
    ds.save('./data/test.json')
    # ds = KpBioDataset.from_encoded('./data/test.json', tokenizer)
    tmp = ds[0]
    print(tmp)