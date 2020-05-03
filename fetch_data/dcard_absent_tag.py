import pandas as pd

df = pd.read_json('dcard.json')
absent_keywords = []
extract_keywords = []

for title, text, topics in zip(df['title'].tolist(),df ['text'].tolist(), df['topics'].tolist()):
    absk = []
    extk = []

    for topic in topics:
        if topic in title or topic in text:
            extk.append(topic)
        else:
            absk.append(topic)
    
    absent_keywords.append(absk)
    extract_keywords.append(extk)

df['absent_keyphrases'] = absent_keywords
df['extract_keyphrases'] = extract_keywords
df = df.drop('topics', axis=1)
df.to_json('dcard_structed.json', force_ascii=False)