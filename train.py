import os
# os.environ['WANDB_MODE'] = 'offline'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
from transformers import TrainingArguments, EarlyStoppingCallback
from model_trainer import ModelTrainer
import numpy as np
import wandb
from models import load_model
from dataset_temperature import load_dataset
from torch.optim import AdamW
from utils import (parse_task,
                   load_config,
                   load_weight,
                   freeze_model,
                   count_parameters,
                   write_json,
                   scatter_plot_with_density,
                   interval_evaluation_opt,
                   interval_evaluation_stability,
                   plot_interval_evaluation,
                   load_metrics,
                   overlap_ratio)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--config', type=str, required=False, default='configs/stability_reda_segment.yaml', help='Path to the YAML config file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    Model, Collator = load_model(config['model']['name'])

    task = parse_task(config['dataset']['name'])
    Dataset = load_dataset(task)
    train_path = os.path.join('data', config['dataset']['name'], config['dataset']['train'])
    val_path = os.path.join('data', config['dataset']['name'], config['dataset']['val'])
    test_path = os.path.join('data', config['dataset']['name'], config['dataset']['test'])
    dataset_train = Dataset(train_path)
    dataset_val = Dataset(val_path)
    dataset_test = Dataset(test_path)

    metrics = load_metrics(task)

    print(f'train: {len(dataset_train)}')
    print(f'val: {len(dataset_val)}')
    print(f'test: {len(dataset_test)}')

    model = Model(config['model'])
    if not task == 'range':
        temp_ranges = dataset_train.default_ranges
        weights = dataset_train.calculate_weights()
        model.loss_fct.set_ranges_and_weights(temp_ranges, weights)
    # transfer learning
    if task == 'range':
        load_weight(model, 'output/split_in_clusters_opt_segment_cw_2_s_1/checkpoint-4224/model.safetensors')

    collate_fn = Collator(config['model']['pretrain_model'])
    try:
        freeze_model(model.pretrain_model)
    except:
        print('No pretrain model to freeze')
    print(f'trainable parameters: {round(count_parameters(model) / 1000000, 2)}M')
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config['training']['lr']))

    wandb.init(project='EnzymeT',
               config={
                   "learning_rate": float(config['training']['lr']),
                   "architecture": config['model']['name'],
                   "dataset": task,
                   "epochs": config['training']['num_epochs'],
               }
               )

    run_name = 'split_in_clusters_' + args.config.split('/')[-1].split('.')[0]

    wandb.run.name = run_name
    print(f'wandb run name: {run_name}')
    args = TrainingArguments(
        output_dir=f'output/{run_name}',
        logging_dir=f'output/{run_name}/log',
        logging_strategy='epoch',
        save_strategy="epoch",
        learning_rate=float(config['training']['lr']),
        per_device_train_batch_size=config['training']['train_batch_size'],
        per_device_eval_batch_size=config['training']['eval_batch_size'],
        num_train_epochs=64,
        weight_decay=float(config['training']['weight_decay']),
        evaluation_strategy="epoch",
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        run_name=wandb.run.name,
        overwrite_output_dir=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=["wandb"],
        fp16=config['training']['fp16'],
        max_grad_norm=config['training']['max_grad_norm'],
        load_best_model_at_end=False if 'segment' in config['model']['name'] else True,
    )

    trainer = ModelTrainer(
        model=model,
        optimizers=(optimizer, None),
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_fn,
        compute_metrics=metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] if not 'segment' in config['model']['name'] else None,
    )

    trainer.train(resume_from_checkpoint=False)

    if not os.path.exists(f'results/{run_name}/'):
        os.makedirs(f'results/{run_name}/')

    datasets = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
    for k, v in datasets.items():
        print(f'------------------------{k}--------------------------')
        predictions, labels, metrics = trainer.predict(v)

        write_json(metrics, f'results/{run_name}/{k}_metrics.json')

        if k == 'test':
            np.save(f'results/{run_name}/predictions.npy', predictions)
            np.save(f'results/{run_name}/labels.npy', labels)

            if task == 'range':
                ratio = overlap_ratio(labels, predictions, save_dir=f'results/{run_name}')
                print(f'overlap ratio: {ratio}')
                scatter_plot_with_density(labels[:, 0], predictions[:, 0], xlabel='experimental temperature values',
                                          ylabel='predicted temperature values', save_dir=f'results/{run_name}',
                                          title='Scatter_plot_for_low_temperature')
                scatter_plot_with_density(labels[:, 1], predictions[:, 1], xlabel='experimental temperature values',
                                            ylabel='predicted temperature values', save_dir=f'results/{run_name}',
                                            title='Scatter_plot_for_high_temperature')
            else:
                scatter_plot_with_density(labels, predictions, xlabel='experimental temperature values',
                                          ylabel='predicted temperature values', save_dir=f'results/{run_name}')

                if task == 'opt':
                    metrics_interval = interval_evaluation_opt(labels, predictions)
                elif task == 'stability':
                    metrics_interval = interval_evaluation_stability(labels, predictions)
                write_json(metrics_interval, f'results/{run_name}/{k}_metrics_interval.json')
                plot_interval_evaluation(metrics_interval, save_dir=f'results/{run_name}')

    wandb.finish()