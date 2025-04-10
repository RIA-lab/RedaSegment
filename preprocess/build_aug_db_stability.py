from augmentations import back_translation_substitute
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
import pandas as pd
import os
import torch
import random


def check_protein(seq):
    for aa in seq:
        if aa not in 'ACDEFGHIKLMNPQRSTVWY':
            return False
    return True

tokenizer = EsmTokenizer.from_pretrained('../esm150')
embedding_model = EsmModel.from_pretrained('../esm150')
embedding_model.to('cuda')


data = pd.read_csv('../data/stability/train.csv')

data_45 = data[data['stability'] < 45]
data45_70 = data[(data['stability'] >= 45) & (data['stability'] < 70)]
data70_100 = data[(data['stability'] >= 70) & (data['stability'] < 100)]
data100_ = data[data['stability'] >= 100]
data_dict = {'db-45': data_45, 'db45-70': data45_70, 'db70-100': data70_100, 'db100-': data100_}

print(len(data_45), len(data45_70), len(data70_100), len(data100_))

data_dict = {'db-45': data_45, 'db45-70': data45_70, 'db70-100': data70_100, 'db100-': data100_}

for k, v in data_dict.items():
    embedding_database = []
    seqs = v['sequence'].tolist()
    for batch in tqdm(range(0, len(seqs), 8)):
        inputs = tokenizer(seqs[batch:batch+8], return_tensors='pt', padding='max_length', truncation=True, max_length=1000)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        outputs = outputs.last_hidden_state
        outputs = torch.mean(outputs, 1)
        embedding_database.append(outputs)
    embedding_database = torch.cat(embedding_database, 0)
    print(embedding_database.shape)
    os.makedirs(f'../data/augmentation_database/stability', exist_ok=True)
    torch.save(embedding_database, f'../data/augmentation_database/stability/{k}.pt')


