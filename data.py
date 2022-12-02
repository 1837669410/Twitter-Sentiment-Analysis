import pandas as pd
import numpy as np
import itertools
import tensorflow as tf
from tqdm import tqdm

sentiment_dict = {
    "Negative": 0,
    "Positive": 1,
    "Neutral": 2,
    "Irrelevant": 3,
}

replace_list = {r"i'm": 'i am',
                r"'re": ' are',
                r"let’s": 'let us',
                r"'s":  ' is',
                r"'ve": ' have',
                r"can't": 'can not',
                r"cannot": 'can not',
                r"shan’t": 'shall not',
                r"n't": ' not',
                r"'d": ' would',
                r"'ll": ' will',
                r"'scuse": 'excuse',
                ',': ' ,',
                '.': ' .',
                '!': ' !',
                '?': ' ?',
                '\s+': ' '}

def clean_text(text):
    text = text.lower()
    for s in replace_list:
        text = text.replace(s, replace_list[s])
    text = ' '.join(text.split())
    return text

def translation_data(corpus, v2i, max_text_length):
    x_train = []
    for c in tqdm(corpus):
        temp = c.split(" ")
        if len(temp) < max_text_length:
            temp += ["UNK"] * (max_text_length - len(temp))
        else:
            temp = temp[:max_text_length]
        for i in range(len(temp)):
            temp[i] = v2i.get(temp[i], 0)
        x_train.append(temp)
    return np.array(x_train)

def load_data(train_path, val_path, words_frequency=5, max_text_length=20):
    train_data = pd.read_csv(train_path, names=['Tweet ID', 'entity', 'sentiment', 'Tweet Content'])
    val_data = pd.read_csv(val_path, names=['Tweet ID', 'entity', 'sentiment', 'Tweet Content'])
    # 1、处理缺失值
    train_data.dropna(inplace=True)
    train_data.index = (range(len(train_data)))
    train_data.info()
    train_data.loc[:, "Tweet Content"].apply(lambda p: clean_text(p))
    print("情绪类别数：\n{}".format(train_data.loc[:, "sentiment"].value_counts()))
    print(train_data.head(5))
    val_data.dropna(inplace=True)
    val_data.index = (range(len(val_data)))
    val_data.info()
    val_data.loc[:, "Tweet Content"].apply(lambda p: clean_text(p))
    # 2、构建字典，将词频数低于5次的词语删除
    words = [c.split(" ") for c in train_data.loc[:,"Tweet Content"]]
    words = np.array(list(itertools.chain(*words)))
    words, words_counts = np.unique(words, return_counts=True)
    fre_index = np.where(words_counts >= words_frequency)
    words, words_counts = words[fre_index], words_counts[fre_index]
    print("词表数：{}".format(len(words_counts)))
    # 3、构建i2v和v2i
    i2v = {i+1:v for i, v in enumerate(words)}
    i2v[0] = "UNK"
    v2i = {v:i for i, v in i2v.items()}
    # 4、将文本转化成可训练格式
    train_corpus = [c for c in train_data.loc[:,"Tweet Content"]]
    val_corpus = [c for c in val_data.loc[:,"Tweet Content"]]
    x_train = translation_data(train_corpus, v2i, max_text_length)
    y_train = np.array([sentiment_dict.get(v) for v in train_data.loc[:,"sentiment"]])
    x_val = translation_data(val_corpus, v2i, max_text_length)
    y_val = np.array([sentiment_dict.get(v) for v in val_data.loc[:,"sentiment"]])
    print("x：\n{}".format(x_train[10:15,:]))
    print("y：\n{}".format(y_train[10:15]))
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(50000).batch(128, drop_remainder=True)
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.shuffle(1000).batch(128, drop_remainder=True)

    return db_train, val_data, len(v2i)

if __name__ == "__main__":
    load_data("twitter_training.csv", "twitter_validation.csv")