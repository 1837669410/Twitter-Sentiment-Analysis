import pandas as pd
import numpy as np
import itertools
import tensorflow as tf
import torch
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

def padding_data(corpus, v2i, max_text_length):
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

def get_ori_data(train_path, val_path, words_frequency):
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
    return train_data, val_data, i2v, v2i

def load_data(train_path, val_path, words_frequency=5, max_text_length=20, framework="tf"):
    # 1、获得数据原始信息
    train_data, val_data, i2v, v2i = get_ori_data(train_path, val_path, words_frequency=words_frequency)
    # 2、将文本转化成可训练格式
    train_corpus = [c for c in train_data.loc[:,"Tweet Content"]]
    val_corpus = [c for c in val_data.loc[:,"Tweet Content"]]
    x_train = padding_data(train_corpus, v2i, max_text_length)
    y_train = np.array([sentiment_dict.get(v) for v in train_data.loc[:,"sentiment"]])
    x_val = padding_data(val_corpus, v2i, max_text_length)
    y_val = np.array([sentiment_dict.get(v) for v in val_data.loc[:,"sentiment"]])
    print("x：\n{}".format(x_train[10:15,:]))
    print("y：\n{}".format(y_train[10:15]))
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    # 3、转化成tf格式的训练数据
    if framework == "tf":
        db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        db_train = db_train.shuffle(50000).batch(128, drop_remainder=True)
        db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        db_val = db_val.shuffle(1000).batch(128, drop_remainder=True)
        return db_train, db_val, len(v2i)
    if framework == "torch":
        x_train, y_train, x_val, y_val = torch.LongTensor(x_train), torch.LongTensor(y_train), torch.LongTensor(x_val), torch.LongTensor(y_val)
        db_train, db_val = torch.utils.data.TensorDataset(x_train, y_train), torch.utils.data.TensorDataset(x_val, y_val)
        db_train = torch.utils.data.DataLoader(db_train, batch_size=128, shuffle=True, drop_last=True)
        db_val = torch.utils.data.DataLoader(db_val, batch_size=128, shuffle=True, drop_last=True)
        return db_train, db_val, len(v2i)