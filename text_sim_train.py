import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
import pdb

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.set_device(0)
device = torch.device("cuda:0")

def preprocess_function(examples, tokenizer, max_length=64):
    return tokenizer(examples['text1'], examples['text2'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

class TextPairDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

def train_model(model_path, data_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    #  {'text1': "...", 'text2': "...", 'label': 0/1}
    dataset = load_dataset('csv', data_files={'train': data_path + '/train_rel.csv', 'test': data_path + '/test_rel.csv'})
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=['text1', 'text2'])

    # train_dataset = TextPairDataset(dataset['train'], dataset['train']['label'])
    # test_dataset = TextPairDataset(dataset['test'], dataset['test']['label'])

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=4,
        # warmup_steps=500,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
    )

    print(f"当前使用设备: {trainer.args.device}")
    print(f"模型所在设备: {next(model.parameters()).device}")

    trainer.train()

    model.save_pretrained(data_path + "/subtitle_sim_bert")
    tokenizer.save_pretrained(data_path + "/subtitle_sim_bert")

if __name__ == "__main__":
    model_path = '../../models/deberta-v2-chinese'
    data_dir = '../dataset'
    dataset = 'cmivqa'
    data_path = os.path.join(data_dir, dataset)
    train_model(model_path, data_path)
