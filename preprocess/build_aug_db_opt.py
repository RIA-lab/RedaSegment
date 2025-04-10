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


data = pd.read_csv('../data/opt/train.csv')
data_25 = data[data['temperature_optimum'] < 25]
data25_50 = data[(data['temperature_optimum'] >= 25) & (data['temperature_optimum'] < 50)]
data50_80 = data[(data['temperature_optimum'] >= 50) & (data['temperature_optimum'] < 80)]
data80_ = data[data['temperature_optimum'] >= 80]
# print(f'<25: {len(data_25)}')
# print(f'25-50: {len(data25_50)}')
# print(f'50-80: {len(data50_80)}')
# print(f'>80: {len(data80_)}')
# aug_num25_50 = len(data25_50) * 0.5
# aug_num_25 = (aug_num25_50 + len(data25_50)-len(data_25))
# aug_num50_80 = (aug_num25_50 + len(data25_50)-len(data50_80))
# aug_num80_ = (aug_num25_50 + len(data25_50)-len(data80_))
# aug_freq25_50 = 0.5
# aug_freq_25 = aug_num_25 / len(data_25)
# aug_freq50_80 = aug_num50_80 / len(data50_80)
# aug_freq80_ = aug_num80_ / len(data80_)
# print(f'25-50: {aug_num25_50}, {aug_freq25_50}')
# print(f'<25: {aug_num_25}, {aug_freq_25}')
# print(f'50-80: {aug_num50_80}, {aug_freq50_80}')
# print(f'>80: {aug_num80_}, {aug_freq80_}')

data_dict = {'db-25': data_25, 'db25-50': data25_50, 'db50-80': data50_80, 'db80-': data80_}
# freq_dict = {'db-25': aug_freq_25, 'db25-50': aug_freq25_50, 'db50-80': aug_freq50_80, 'db80-': aug_freq80_}

all_rows = []
for k, v in data_dict.items():
    # freq = freq_dict[k]
    # aug_rows = []
    # #iterating rows over the dataframes
    # for index, row in v.iterrows():
    #     seq = row['sequence']
    #     if not check_protein(seq):
    #         continue
    #     if freq < 1:
    #         flag = random.random() < freq
    #         if not flag:
    #             continue
    #         else:
    #             aug_seq = back_translation_substitute(list(seq), 0.0125)
    #             #duplicate the row
    #             new_row = row.copy()
    #             new_row['sequence'] = ''.join(aug_seq)
    #             aug_rows.append(new_row)
    #     else:
    #         for _ in range(int(freq)):
    #             aug_seq = back_translation_substitute(list(seq), 0.0125)
    #             new_row = row.copy()
    #             new_row['sequence'] = ''.join(aug_seq)
    #             aug_rows.append(new_row)
    # print(f'{k}: {len(aug_rows)}')
    # all_rows.extend(aug_rows)

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
    os.makedirs(f'../data/augmentation_database/opt', exist_ok=True)
    torch.save(embedding_database, f'../data/augmentation_database/opt/{k}.pt')

# #merge the augmented data
# all_rows = pd.DataFrame(all_rows)
# all_rows.to_csv('../data/augmentation_database/opt/augmented.csv', index=False)
# #merge all_rows with the original data
# all_data = pd.concat([data, all_rows])
# all_data.to_csv('../data/opt/train_aug.csv', index=False)
