from augmentations import back_translation_substitute
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
import pandas as pd
import os
import torch


tokenizer = EsmTokenizer.from_pretrained('../esm150')
embedding_model = EsmModel.from_pretrained('../esm150')
embedding_model.to('cuda')


data = pd.read_csv('../data/range/train.csv')
data_20 = data[data['temperature_low'] < 20]
data20_40 = data[(data['temperature_low'] >= 20) & (data['temperature_low'] < 40)]
data40_60 = data[(data['temperature_low'] >= 40) & (data['temperature_low'] < 60)]
data60_80 = data[(data['temperature_low'] >= 60) & (data['temperature_low'] < 80)]
data80_ = data[data['temperature_low'] >= 80]

print(f'<20: {len(data_20)}')
print(f'20-40: {len(data20_40)}')
print(f'40-60: {len(data40_60)}')
print(f'60-80: {len(data60_80)}')
print(f'>80: {len(data80_)}')

data_dict = {'db-20': data_20, 'db20-40': data20_40, 'db40-60': data40_60, 'db60-80': data60_80, 'db80-': data80_}

for k, v in data_dict.items():
    embedding_database = []
    seqs = v['sequence'].tolist()
    for batch in tqdm(range(0, len(seqs), 32)):
        inputs = tokenizer(seqs[batch:batch+32], return_tensors='pt', padding='max_length', truncation=True, max_length=1000)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        outputs = outputs.last_hidden_state
        outputs = torch.mean(outputs, 1)
        embedding_database.append(outputs)
    embedding_database = torch.cat(embedding_database, 0)
    print(embedding_database.shape)
    os.makedirs(f'../data/augmentation_database/range/low', exist_ok=True)
    torch.save(embedding_database, f'../data/augmentation_database/range/low/{k}.pt')

data_40 = data[data['temperature_high'] < 40]
data40_60 = data[(data['temperature_high'] >= 40) & (data['temperature_high'] < 60)]
data60_80 = data[(data['temperature_high'] >= 60) & (data['temperature_high'] < 80)]
data80_100 = data[(data['temperature_high'] >= 80) & (data['temperature_high'] < 100)]
data100_ = data[data['temperature_high'] >= 100]

print(f'<40: {len(data_40)}')
print(f'40-60: {len(data40_60)}')
print(f'60-80: {len(data60_80)}')
print(f'80-100: {len(data80_100)}')
print(f'>100: {len(data100_)}')

data_dict = {'db-40': data_40, 'db40-60': data40_60, 'db60-80': data60_80, 'db80-100': data80_100, 'db100-': data100_}


for k, v in data_dict.items():
    embedding_database = []
    seqs = v['sequence'].tolist()
    for batch in tqdm(range(0, len(seqs), 32)):
        inputs = tokenizer(seqs[batch:batch+32], return_tensors='pt', padding='max_length', truncation=True, max_length=1000)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        outputs = outputs.last_hidden_state
        outputs = torch.mean(outputs, 1)
        embedding_database.append(outputs)
    embedding_database = torch.cat(embedding_database, 0)
    print(embedding_database.shape)
    os.makedirs(f'../data/augmentation_database/range/high', exist_ok=True)
    torch.save(embedding_database, f'../data/augmentation_database/range/high/{k}.pt')
