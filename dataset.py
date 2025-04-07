import torch
from torch.utils.data import Dataset
import pdb


class VideoDataset(Dataset):
    def __init__(self, input_data, tokenizer, max_length=128, get_round=3):
        self.input_data = input_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        data_item = self.input_data[idx]
        question = data_item[0][0]
        question_desc = data_item[0][1]
        subtitle_desc = data_item[0][2]
        subtitle_context = data_item[0][3]
        clip_feature_id = data_item[0][4]
        start = data_item[0][5]
        duration = data_item[0][6]
        
        labels_relevancy = data_item[1][0]
        labels_event = data_item[1][1]

        context_list = subtitle_context.split('|')
        input_context = '[SEP]'.join(context_list)

        question = f"Question: {question}"
        question_with_desc = f"{question} Description: {question_desc}"
        subtitle = f"Subtitle: {subtitle_desc}"
        subtitle_with_context = f"Subtitle: {subtitle_desc} Context: {input_context}"

        input_question = self.tokenizer.encode(question, padding='max_length', truncation=True, max_length=round(self.max_length / 8))
        input_question_deep = self.tokenizer.encode(question_with_desc, padding='max_length', truncation=True, max_length=round(self.max_length / 2))
        input_subtitle_desc = self.tokenizer.encode(subtitle, padding='max_length', truncation=True, max_length=round(self.max_length / 8))
        input_subtitle_context = self.tokenizer.encode(subtitle_with_context, padding='max_length', truncation=True, max_length=round(self.max_length))

        return {
            "question": question,
            "clip_feature_id": clip_feature_id,

            "input_ids_question": torch.tensor(input_question),
            "input_ids_question_deep": torch.tensor(input_question_deep),
            "input_ids_subtitle_desc": torch.tensor(input_subtitle_desc),
            "input_ids_subtitle_context": torch.tensor(input_subtitle_context),

            "labels_relevancy": torch.tensor(labels_relevancy, dtype=torch.long),
            "labels_event": torch.tensor(labels_event, dtype=torch.long),

            "start": torch.tensor(start, dtype=torch.float),
            "duration": torch.tensor(duration, dtype=torch.float),
        }
