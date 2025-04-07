from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import random
import math
import numpy as np
from tqdm import tqdm
from model import RCVTA
from dataset import VideoDataset
from data_process import *
from config import *
from evaluate import *
import pdb

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def adjust_learning_rate(param_group, LR, epoch, warmup_epochs, num_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 1e-6
    if epoch < warmup_epochs:
        lr = max(LR * epoch / warmup_epochs, min_lr)
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    param_group["lr"] = lr
    return lr

def print_settings(args):
    if args.use_dialogue and args.use_rag and args.use_rewrite_dialogue and args.use_rewrite_subtitle:
        print("====================================")
        print("Conversation Rewrite + Subtitle Retrieval + Subtitle Rewrite")
        print("====================================")
    elif args.use_dialogue and args.use_rewrite_dialogue and args.use_rewrite_subtitle and not args.use_rag:
        print("====================================")
        print("Conversation Rewrite + Subtitle Rewrite")
        print("====================================")
    elif args.use_dialogue and args.use_rag and args.use_rewrite_subtitle and not args.use_rewrite_dialogue:
        print("====================================")
        print("Conversation + Subtitle Retrieval + Subtitle Rewrite")
        print("====================================")
    elif args.use_rag and args.use_rewrite_subtitle and not args.use_dialogue:
        print("====================================")
        print("Subtitle Retrieval + Subtitle Rewrite")
        print("====================================")
    elif args.use_dialogue and args.use_rag and args.use_rewrite_dialogue and not args.use_rewrite_subtitle:
        print("====================================")
        print("Conversation Rewrite + Subtitle Retrieval")
        print("====================================")
    elif not args.use_rag and not args.use_dialogue and not args.use_rewrite_subtitle:
        print("====================================")
        print("No strategy")
        print("====================================")
    else:
        raise ValueError("Unsupported setting values")

def run():
    args = set_argument()
    print_settings(args)
    subtitle_dir = os.path.join(args.dataset_dir, args.dataset_name, 'video_process', 'subtitles')
    clip_feature_dir = os.path.join(args.dataset_dir, args.dataset_name, 'video_process', 'clipfeatures')
    train_dialogue_path = os.path.join(args.dataset_dir, args.dataset_name, 'train_dialogue_desc.json')
    test_dialogue_path = os.path.join(args.dataset_dir, args.dataset_name, 'test_dialogue_desc.json')
    train_data_path = os.path.join(args.dataset_dir, args.dataset_name, 'train.json')
    test_data_path = os.path.join(args.dataset_dir, args.dataset_name, 'test.json')
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)

    print("Loading input data...")
    input_data_train = load_data(
            args.dataset_dir, 
            args.dataset_name, 
            "train", 
            data_ratio = args.train_ratio,
            subtitle_ratio=args.subtitle_ratio, 
            use_dialogue=args.use_dialogue,
            dialogue_round=args.dialogue_round,
            use_rag=args.use_rag,
            use_rewrite_dialogue=args.use_rewrite_dialogue,
            use_rewrite_subtitle=args.use_rewrite_subtitle
        )
    input_data_test = load_data(
            args.dataset_dir, 
            args.dataset_name, 
            "test", 
            subtitle_ratio=args.subtitle_ratio, 
            use_dialogue=args.use_dialogue,
            dialogue_round=args.dialogue_round,
            use_rag=args.use_rag,
            use_rewrite_dialogue=args.use_rewrite_dialogue,
            use_rewrite_subtitle=args.use_rewrite_subtitle
        )
        
    dataset_train = VideoDataset(
            input_data_train, 
            tokenizer, 
            max_length=args.max_length
        )
    dataset_test = VideoDataset(
            input_data_test, 
            tokenizer, 
            max_length=args.max_length
        )

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, drop_last=False, pin_memory=True, shuffle=False)

    device = torch.device(args.cuda_name if torch.cuda.is_available() and args.use_cuda else "cpu")

    model = RCVTA(args, device)
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    trainable_params, all_param = model.print_trainable_params()
    print("Trainable parameters: "+str(trainable_params)+" All parameters: "+str(all_param))
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        step = 0
        progress_bar = tqdm(range(len(dataloader_train)))
        for i, batch in enumerate(dataloader_train):

            input_ids_question = batch["input_ids_question"].to(device)
            input_ids_question_deep = batch["input_ids_question_deep"].to(device)
            input_ids_subtitle_desc = batch["input_ids_subtitle_desc"].to(device)
            input_ids_subtitle_context = batch["input_ids_subtitle_context"].to(device)

            clip_feature_ids = batch["clip_feature_id"]

            labels_rel = batch["labels_relevancy"].to(device)
            labels_evt = batch["labels_event"].to(device)

            loss, pos_index = model(
                input_ids_question,
                input_ids_question_deep,
                input_ids_subtitle_desc,
                input_ids_subtitle_context, 
                clip_feature_ids,
                labels_rel,
                labels_evt
            )

            print("step {}, current loss: {}".format(step+1, loss))
            print("positive index: {}".format(str(pos_index)))

            optimizer.zero_grad()
            loss.backward()
            adjust_learning_rate(optimizer.param_groups[0], args.lr, i / len(dataloader_train) + epoch, args.warmup_epochs, args.epochs)
            print("Learning rate: " + str(optimizer.param_groups[0]["lr"]))
            optimizer.step()

            total_loss += loss.item()
            step += 1
            progress_bar.update(1)
            # print("Test evaluation")
            # evaluate(model, dataloader_test, test_dialogue_path, args.batch_size, device)

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader_train):.4f}")
        print("Evaluate on test dataset...")
        evaluate(model, dataloader_test, test_dialogue_path, args.batch_size, device)

if __name__ == '__main__':
    run()

