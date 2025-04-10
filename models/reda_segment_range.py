import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers import EsmModel, EsmTokenizer
import torch.nn.functional as F
import os
from models.loss_func import TemperatureRangeLoss


class SearchLayer(nn.Module):
    def __init__(self, initial_weight, idx, num_labels):
        super(SearchLayer, self).__init__()
        self.num_labels = num_labels
        weight = initial_weight
        weight = weight / weight.norm(dim=-1, keepdim=True)
        weight = weight.T
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        self.embed_dim = weight.shape[0]
        self.id = F.one_hot(torch.tensor(idx, dtype=torch.int64), num_classes=num_labels)

    def forward(self, queries):
        queries_norm = queries / queries.norm(dim=-1, keepdim=True)
        similarities = torch.matmul(queries_norm, self.weight)
        similarity, index = torch.max(similarities, dim=-1)
        CLS = self.id.to(queries.device) * similarity.unsqueeze(-1).repeat(1, self.num_labels)
        return CLS

class MultiScaleSamplingLayer(nn.Module):
    def __init__(self, scales, embedding_dim):
        super().__init__()
        self.samplingLayers = nn.ModuleList([nn.Conv1d(embedding_dim, embedding_dim, 2, 2) for _ in range(scales)])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        multi_scale_inputs = [x]
        for layer in self.samplingLayers:
            x = layer(x)
            multi_scale_inputs.append(x)
        multi_scale_inputs = [_.permute(0, 2, 1) for _ in multi_scale_inputs]
        return multi_scale_inputs


class SequenceSegmentation(nn.Module):
    def __init__(self, segment_lengths):
        super().__init__()
        self.segment_lengths = segment_lengths

    def forward(self, x):
        for idx, length in enumerate(self.segment_lengths):
            x[idx] = x[idx].reshape(x[idx].shape[0], -1, length, x[idx].shape[-1])
        return x


class MultiScaleSegmentConv2d(nn.Module):
    def __init__(self, kernel_size, segment_lengths, embedding_dim):
        super().__init__()
        self.segment_lengths = segment_lengths
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(embedding_dim, embedding_dim, (kernel_size, _), stride=(1, 1), padding=(2, 0)) for _ in
             segment_lengths])

    def forward(self, x):
        images = [layer(_.permute(0, 3, 1, 2)) for _, layer in zip(x, self.conv_layers)]
        images = [torch.squeeze(_.permute(0, 2, 3, 1), dim=2) for _ in images]
        return images


class ColRowSegmentAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, attention_range):
        super().__init__()
        self.col_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.row_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.attention_range = attention_range

    def forward(self, x, logits):
        logits = logits.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = x + logits
        x = x.reshape(x.shape[0], -1, self.attention_range, x.shape[-1])
        x_col = x.reshape(-1, x.shape[1], x.shape[-1])
        x_row = x.reshape(-1, x.shape[2], x.shape[-1])
        x_col = self.col_attention(x_col)
        x_row = self.row_attention(x_row)
        x = x_col.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[-1]) + x_row.reshape(x.shape[0], x.shape[1],
                                                                                           x.shape[2], x.shape[-1])
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


class MultiScaleSegmentAttention(nn.Module):
    def __init__(self, scales, embedding_dim, num_heads, num_layers, attention_range):
        super().__init__()
        self.segment_attentions = nn.ModuleList(
            [ColRowSegmentAttention(embedding_dim, num_heads, num_layers, attention_range) for _ in range(scales + 1)])

    def forward(self, x, logits):
        for idx, layer in enumerate(self.segment_attentions):
            x[idx] = layer(x[idx], logits)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_layer1 = nn.Linear(input_dim, input_dim)
        self.attention_layer2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_scores = self.attention_layer1(x)
        attn_scores = torch.tanh(attn_scores)
        attn_scores = self.attention_layer2(attn_scores).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled_output = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return pooled_output, attn_weights


class MultiScalePooling(nn.Module):
    def __init__(self, scales, embedding_dim):
        super().__init__()
        self.poolingLayers = nn.ModuleList([AttentionPooling(embedding_dim) for _ in range(scales + 1)])

    def forward(self, x):
        pooled_outputs = []
        for idx, layer in enumerate(self.poolingLayers):
            pooled_output, attn_weights = layer(x[idx])
            pooled_outputs.append((pooled_output, attn_weights))
        return pooled_outputs


class MergeRegressionHead(nn.Module):
    def __init__(self, scales, input_dim):
        super().__init__()
        self.regression_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(scales + 1)])

    def forward(self, x, logits):
        multi_scale_pred = [layer(_+logits) for _, layer in zip(x, self.regression_layers)]
        multi_scale_pred = torch.cat(multi_scale_pred, dim=1)
        pred_max = torch.max(multi_scale_pred, 1)[0]
        pred_min = torch.min(multi_scale_pred, 1)[0]
        pred = torch.mean(multi_scale_pred, 1)
        return pred, pred_min, pred_max


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pretrain_model = EsmModel.from_pretrained(config['pretrain_model'])
        config_db_low = config['database_low']
        self.search_layers_low = nn.ModuleList(
            [SearchLayer(torch.load(os.path.join(config_db_low['path_dir'], _)), idx, config['num_labels']) for idx, _ in
             enumerate(config_db_low['file_list'])])
        config_db_high = config['database_high']
        self.search_layers_high = nn.ModuleList(
            [SearchLayer(torch.load(os.path.join(config_db_high['path_dir'], _)), idx, config['num_labels']) for idx, _ in
             enumerate(config_db_high['file_list'])])
        self.logits_proj = nn.Linear(config['num_labels']*2, config['embed_dim'])
        self.logits_low_proj = nn.Linear(config['num_labels'], config['embed_dim'])
        self.logits_high_proj = nn.Linear(config['num_labels'], config['embed_dim'])
        self.sample_layer = MultiScaleSamplingLayer(config['scales'], config['embed_dim'])
        self.segmentation = SequenceSegmentation(config['segment_lengths'])
        self.segment_conv2d = MultiScaleSegmentConv2d(config['kernel_size'], config['segment_lengths'],
                                                      config['embed_dim'])
        self.segment_attention = MultiScaleSegmentAttention(config['scales'], config['embed_dim'], config['num_heads'],
                                                            config['num_layers'], config['attention_range'])
        self.segment_pooling_low = MultiScalePooling(config['scales'], config['embed_dim'])
        self.segment_pooling_high = MultiScalePooling(config['scales'], config['embed_dim'])
        self.regression_low = MergeRegressionHead(config['scales'], config['embed_dim'])
        self.regression_high = MergeRegressionHead(config['scales'], config['embed_dim'])
        self._keys_to_ignore_on_save = []
        self.loss_fct = TemperatureRangeLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            outputs = self.pretrain_model(input_ids, attention_mask)

        hidden_state = outputs.last_hidden_state
        queries = torch.mean(hidden_state, 1)

        logits_low = [layer(queries) for layer in self.search_layers_low]
        logits_low = torch.sum(torch.stack(logits_low), dim=0)
        # scaled softmax
        logits_low = torch.softmax(logits_low * 100, dim=-1)

        logits_high = [layer(queries) for layer in self.search_layers_high]
        logits_high = torch.sum(torch.stack(logits_high), dim=0)
        # scaled softmax
        logits_high = torch.softmax(logits_high * 100, dim=-1)

        logits = torch.cat([logits_low, logits_high], dim=1)
        logits = self.logits_proj(logits)
        logits_low = self.logits_low_proj(logits_low)
        logits_high = self.logits_high_proj(logits_high)

        hidden_state = self.sample_layer(hidden_state)
        hidden_state = self.segmentation(hidden_state)
        hidden_state = self.segment_conv2d(hidden_state)
        hidden_state = self.segment_attention(hidden_state, logits)

        hidden_state_low = self.segment_pooling_low(hidden_state)
        hidden_state_low, attn_weights_low = zip(*hidden_state_low)
        attn_weights_low = torch.mean(torch.stack(attn_weights_low), 0)
        pred_low, pred_low_min, pred_low_max = self.regression_low(hidden_state_low, logits_low)

        hidden_state_high = self.segment_pooling_high(hidden_state)
        hidden_state_high, attn_weights_high = zip(*hidden_state_high)
        attn_weights_high = torch.mean(torch.stack(attn_weights_high), 0)
        pred_high, pred_high_min, pred_high_max = self.regression_high(hidden_state_high, logits_high)

        pred = torch.stack([pred_low, pred_high], dim=1)
        loss = None
        if labels is not None:
            loss = self.loss_fct(pred, labels)
        return ModelOutput(loss=loss, pred=pred)


class Collator:
    def __init__(self, pretrain_model):
        self.tokenizer = EsmTokenizer.from_pretrained(pretrain_model)

    def __call__(self, batch):
        seqs = [_.sequence for _ in batch]
        inputs = self.tokenizer(list(seqs), return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        labels_low = [_.labels_low for _ in batch]
        labels_low = torch.tensor(labels_low).float()
        labels_high = [_.labels_high for _ in batch]
        labels_high = torch.tensor(labels_high).float()
        inputs['labels'] = torch.stack([labels_low, labels_high], dim=1)
        return inputs

