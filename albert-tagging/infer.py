import os

import torch
import torch.nn as nn
from tqdm import tqdm

from model import AlbertCrf


class KeyphrasePredictor:
    def __init__(self, tokenizer, bert_path='./albert_base', ckpt='./ckpt', maxlen=512, num_classes=3):
        self.tokenizer = tokenizer
        self.maxlen = maxlen

        self.model = AlbertCrf(num_classes, model_path=bert_path)
        self.model.load_state_dict(torch.load(ckpt))
        self.model = self.model.cuda()
        self.model.eval()

    def predict(self, text):
        inp_ids, seg_ids, mask_ids = self.encode(text)
        inp_ids, seg_ids, mask_ids = inp_ids.unsqueeze(0).cuda(
        ), seg_ids.unsqueeze(0).cuda(), mask_ids.unsqueeze(0).cuda()

        hidden = self.model(inp_ids, seg_ids, mask_ids)
        path_score, best_path = self.model.crf(hidden, mask_ids)
        best_path = best_path.squeeze(0).detach().cpu()

        ans = []
        for idx in range(len(best_path)):
            if best_path[idx] == 1:
                word = []
                while best_path[idx] != 0:
                    word.append(text[idx - 1])
                    idx += 1
                ans.append(''.join(word))
        return ans
    

    def encode(self, text: str):
        inp = self.tokenizer.encode_plus(
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
        return torch.tensor(inp_ids), torch.tensor(type_ids), torch.tensor(attention_mask)


if __name__ == '__main__':
    from data import KpBioDataset
    from transformers import BertTokenizerFast
    from torch.utils.data import DataLoader

    BATCH_SIZE = 8 * 8
    tokenizer = BertTokenizerFast.from_pretrained('./albert_base')
    
    text = '''跨國文化的國家在歐洲不同國家待了快10年 就在今年簽證完結之後回國了 本以為回來是開心的 終於喝到每天念掛的珍奶跟日食 頭1 2個月在找工作還有跟朋友團聚然後突然爆發疫症 就在這個待業期間 想慢慢適應這一切 每天也在想這到底是我想待到養老的國家嗎 畢竟自己心裡是個華人 但是習慣了西方的生活方式 家人朋友也說我太獨立 已經不太合群 之前在英國住過    '''

    kp = KeyphrasePredictor(tokenizer, './albert_base',
                            ckpt='ckpt/step_8502.ckpt')
    print(kp.predict(text))
