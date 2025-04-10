import numpy as np
import yaml
from utils import load_weight
from models import load_model
import torch
import os
from dataclasses import dataclass
import pandas as pd
from dataset_temperature import DatasetMutation
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


@dataclass
class InfernceSeq:
    accession: str
    sequence: str
    label: float = 0


def shrink_range(pred_min, pred_max, target_range=10):
    """
    Shrinks the range proportionally while keeping the ratio of pred_min and pred_max.

    Args:
        pred_min (float): The minimum predicted value.
        pred_max (float): The maximum predicted value.
        target_range (float): The desired range width.

    Returns:
        tuple: Adjusted (pred_min, pred_max).
    """
    # Compute the current range
    current_range = pred_max - pred_min
    if current_range <= target_range:
        return pred_min, pred_max  # No need to adjust if already within the range

    # Compute the scaling factor
    scale_factor = target_range / current_range

    # Shrink pred_min and pred_max proportionally
    pred_center = (pred_min + pred_max) / 2
    new_pred_min = pred_center - (pred_center - pred_min) * scale_factor
    new_pred_max = pred_center + (pred_max - pred_center) * scale_factor

    return new_pred_min, new_pred_max


class InferenceModel:
    def __init__(self, config_path, weight_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        Model, Collator = load_model(config['model']['name'])
        self.collate_fn = Collator(config['model']['pretrain_model'])
        self.model = Model(config['model'])
        load_weight(self.model, weight_path)
        self.model.to('cuda')
        self.model.inference = True
        self.model.eval()

    def inference(self, data):
        inputs = self.collate_fn(data)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            # print(outputs)
        for idx, item in enumerate(data):
            inference_accession = item.accession
            inference_seq = item.sequence

            if not os.path.exists(f'inference_results/{inference_accession}'):
                os.mkdir(f'inference_results/{inference_accession}')
            with open(f'inference_results/{inference_accession}/prediction.txt', 'w') as file:
                file.write(f'accession: {inference_accession}\n')
                file.write(f'sequence: {inference_seq}\n')
                for k, v in outputs.items():
                    if k == 'loss':
                        continue
                    elif k == 'pred_min':
                        pred_min = v[idx].cpu().numpy()
                    elif k == 'pred_max':
                        pred_max = v[idx].cpu().numpy()
                    elif k == 'pred':
                        pred = v[idx].cpu().numpy()
                    elif 'attn' in k:
                        v = v[idx].cpu().numpy().tolist()
                        v = np.repeat(v, 8)
                        v = v[:len(inference_seq)]
                        attn_weights = pd.DataFrame({'attn_weights': v})
                        attn_weights.to_csv(f'inference_results/{inference_accession}/{k}.csv', index=False)

                pred_min, pred_max = shrink_range(pred_min, pred_max)
                file.write(f'pred_min: {pred_min}\n')
                file.write(f'pred_max: {pred_max}\n')
                file.write(f'pred: {pred}\n')

        return outputs.pred.cpu().numpy()

    def mutation_inference(self, data):
        for item in data:
            baseline = self.inference([item])
            inference_accession = item.accession
            inference_seq = item.sequence
            attn_weights = pd.read_csv(f'inference_results/{inference_accession}/attn_weights.csv')
            attn_weights = attn_weights['attn_weights'].values
            attn_weights = attn_weights * 600
            dataset = DatasetMutation(inference_seq)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1, collate_fn=self.collate_fn)

            preds = []
            for batch in tqdm(dataloader):
                batch = {k: v.to('cuda') for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                preds.extend(outputs.pred.cpu().numpy().tolist())

            inference_result = dataset.data.copy()
            inference_result['pred'] = preds
            inference_result['pred'] = inference_result['pred'] - baseline
            inference_result['effect'] = inference_result['pred'].apply(lambda x: x * 50)
            inference_result.drop(columns=['sequence', 'label', 'pred'], inplace=True)
            inference_result = inference_result[inference_result['mutation'] != '-']
            amino_acid = 'ACDEFGHIKLMNPQRSTVWY'
            mutation_map = pd.DataFrame({_ : [0.0 for _ in range(len(inference_seq))] for _ in amino_acid})
            mutation_map.insert(0, 'weight', attn_weights)
            for _, row in inference_result.iterrows():
                mutation = row['mutation']
                pos = int(mutation[1:-1]) - 1
                mutation_map.loc[pos, mutation[-1]] = row['effect']
            mutation_map.fillna(0, inplace=True)
            #plot and save the heatmap
            plt.figure(figsize=(len(inference_seq) // 3, 6))
            #transpose the matrix to make the plot more readable
            sns.heatmap(mutation_map.transpose(),
                        cmap='coolwarm',
                        linewidths=0.1,
                        xticklabels=[_+str(i+1) for i, _ in enumerate(inference_seq)],
                        cbar_kws={'pad': 0.01}
                        )
            plt.xticks(rotation=90, fontsize=8)
            plt.yticks(fontsize=10)
            plt.xlabel("Position in Sequence", fontsize=12)
            plt.ylabel("Amino Acid", fontsize=12)
            plt.tight_layout()
            plt.savefig(f'inference_results/{inference_accession}/mutation_map.png', dpi=300, bbox_inches="tight")
            mutation_map['seq'] = [_ for _ in inference_seq]
            if not os.path.exists(f'inference_results/{inference_accession}'):
                os.mkdir(f'inference_results/{inference_accession}')
            inference_result.to_csv(f'inference_results/{inference_accession}/mutation_inference.csv', index=False)
            mutation_map.to_csv(f'inference_results/{inference_accession}/mutation_map.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model inference on CSV input.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    parser.add_argument('--weight', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--file', type=str, required=True, help='CSV file containing enzyme sequences')

    args = parser.parse_args()

    # Load model
    model = InferenceModel(args.config, args.weight)

    # Read CSV file
    data = pd.read_csv(args.file)
    inference_data = [InfernceSeq(row['accession'], row['sequence']) for _, row in data.iterrows()]

    # Run inference
    model.inference(inference_data)
    model.mutation_inference(inference_data)

    # python
    # inference.py - -config
    # configs / stability_reda_segment_cw_2_s_1.yaml - -weight
    # output / split_clusters_stability_reda_segment_cw_2_s_1 / checkpoint - 1408 / model.safetensors - -file
    # enzyme_seq.csv
