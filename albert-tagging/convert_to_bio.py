from typing import List
import re



def remove_special_char(text: str):
    # text = re.sub(
    #     r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
    # # remove sent from ...
    # text = text.split('--\nSent ')[0]
    # keep only eng, zh, number
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    text = rule.sub(' ', text)
    text = ' '.join(text.split())
    return text


def convert_to_bio_tags(text: str, extract_keywords: List[str]):
    bio = ['O'] * len(text)
    for keyword in extract_keywords:
        pos = text.find(keyword)
        while pos != -1:
            bio[pos] = 'B'
            for idx in range(1, len(keyword)):
                bio[pos + idx] = 'I'
            pos = text.find(keyword, pos + len(keyword))

    return bio


if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    from sklearn.utils import shuffle

    FILE = 'test.txt'

    df = pd.read_json('dcard_structed.json')
    df = shuffle(df)

    text = (df['title'] + df['text']).to_list()[140000:]
    keywords = df['extract_keyphrases'].to_list()[140000:]
    
    with open(FILE, 'w') as f:
        for t, k in zip(text, keywords):
            t = remove_special_char(t)
            bio = convert_to_bio_tags(t, k)
            f.write(t + '\n')
            f.write(''.join(bio) + '\n')
