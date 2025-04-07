import os
import torch
import json
import pdb
from transformers import AutoTokenizer, AutoModelForMaskedLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_id = "/data/users/bitlab/models/modern-bert-base"
device = "cuda:2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(
        model_id, 
        low_cpu_mem_usage=True,
        output_hidden_states=True,
        return_dict_in_generate=True).to(device)

subtitle_path = "/data/users/zc/dataset/mm-medical/video_process/subtitles"
subtitle_feature_path = "/data/users/zc/dataset/mm-medical/video_process/subtitle_features"
vid_list = os.listdir(subtitle_path)
if '.DS_Store' in vid_list:
    vid_list.remove('.DS_Store')
vids = sorted(vid_list)
for vid in vids:
    subtitle_file = os.path.join(subtitle_path, vid, vid+'.json')
    with open(subtitle_file, 'r') as f:
        subtitle_content_list = json.load(f)
        subtitle_content = ' '.join([item['text'] for item in subtitle_content_list])
    inputs = tokenizer([subtitle_content], max_length=8192, padding=False, truncation=True, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    subtitle_feature_vid = os.path.join(subtitle_feature_path, vid)
    cls_embed = outputs.hidden_states[-1][:, 0, :]
    subtitle_feature_file = os.path.join(subtitle_feature_path, vid, vid+'.pt')
    if not os.path.exists(subtitle_feature_vid):
        os.mkdir(subtitle_feature_vid)
        torch.save(cls_embed, subtitle_feature_file)
    else:
        if not os.path.exists(subtitle_feature_file):
            torch.save(cls_embed, subtitle_feature_file)
        else:
            continue
    print("Processed video " + vid)
