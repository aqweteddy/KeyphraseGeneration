# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from pymongo import MongoClient
import pandas as pd


# cli = MongoClient('mongodb://user:1234@linux.cs.ccu.edu.tw:27018')
# cur = cli['forum']['dcard']
# df = pd.DataFrame(list(cur.find({'forum': 'dcard'}, {'_id': False, 'title': True, 'text': True, 'board': True, 'topics': True})))


# # %%
# topics = df['topics'].to_list()

# mask = []
# for topic in topics:
#     mask.append(0 if not topic else 1)


# # %%
# df.head()


# # %%
# import numpy as np
# df = df[np.array(mask, dtype=np.bool)]


# # %%
# print(len(df))


# # %%
# df.to_json('dcard.json')


# %%
df = pd.read_json('dcard_structed.json')
# df.index = ['']


# %%

import re

def remove_special_char(text: str):
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
    # remove sent from ... 
    text = text.split('--\nSent ')[0]
    # keep only eng, zh, number
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    text = rule.sub(' ', text)
    text = ' '.join(text.split())
    return text


# %%
remove_special_char(' sdjf      <<<           kjlk')


# %%
from gaisTokenizer import Tokenizer
# from ckiptagger import WS
from tqdm import tqdm
import json
# ws = WS("./data", disable_cuda=False)

ws = Tokenizer(token='En596151595E7D5C555A62575C5C7A565C0D27110E59103A1427193A5B240715530F651B03262C3D05261C195B54AEF1E88EC2E28FD0EC560uEn596151595E7D5C555A62575C5C7A565C0D27110E59103A1427193A5B240715530F651B03262C3D05261C195B54AEF1E88EC2E28FD0EC560u')

ori_title = []
raw_text = []
raw_title = []
seg_text = []
seg_title = []
text, title = [], []
n = 0
for ti, te in zip(tqdm(df['title'].tolist()), df['text'].tolist()):
    ori_title.append(ti)
    raw_text.append(remove_special_char(te))
    raw_title.append(remove_special_char(ti))

    title = ws.tokenize(raw_title[-1])
    text = ws.tokenize(raw_text[-1])
    seg_text.append(text)
    seg_title.append(title)

    if (n+1) %  1000 == 0:
        tmp = pd.DataFrame({'title': ori_title, 'raw_title': raw_title, 'raw_text': raw_text, 'seg_title': seg_title, 'seg_text': seg_text})
        tmp.to_json('tmp.json')
    n += 1


title = ws.tokenize(raw_title[-1])
text = ws.tokenize(raw_text[-1])
seg_text.append(text)
seg_title.append(title)

# %%
print(len(seg_text), len(df['text']))
df['seg_text'] = seg_text
df['seg_title'] = seg_title
df['raw_text'] = raw_text
df['raw_title'] = raw_title
df.to_json('dcard_processed.json')

# %%


