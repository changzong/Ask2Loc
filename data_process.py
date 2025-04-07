import torch
import importlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import json
import random
import os
import csv
import pickle
import openai
from tqdm import tqdm
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from typing import Optional, cast
import numpy as np
import numpy.typing as npt
import pdb
from prompts import relevance_scoring_prompt, further_questioning_prompt, yes_no_answering_prompt_2, subtitle_rewrite_prompt, subtitle_rewrite_prompt_zh, further_questioning_prompt_zh, yes_no_answering_prompt_2_zh, dialogue_summary_prompt, dialogue_summary_prompt_zh
from config import set_argument
from sub_rewrite_llm import SubRewrite


def model_inference_api(sys_input, user_input, model_name, api_key):
    openai.api_key = api_key
    res = openai.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": sys_input},
            {"role": "user", "content": user_input},
        ],
    )
    output = res.choices[0].message.content
    return output

def model_inference_llama(sys_input, user_input, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": sys_input},
        {"role": "user", "content": user_input}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id = tokenizer.eos_token_id,
    )

    response = outputs[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(response, skip_special_tokens=True)
    return generated_text

# generate scores for non-ground-truth descriptions
def score_generate(data_path, desc_dir, score_path, args):
    question_desc_score_list = []
    processed_id = []
    if os.path.exists(score_path):
        with open(score_path, 'r') as f:
            question_desc_score_list = json.load(f)
            processed_id = [item['sample_id'] for item in question_desc_score_list]
    print('Already processed ' + str(len(processed_id)))
    with open(data_path, 'r') as f1:
        data = json.load(f1)
    for i in tqdm(range(len(data))):
        item = data[i]
        vid = item['video_id']
        question = item['question']
        answer_start = item['answer_start_second']
        answer_end = item['answer_end_second']
        desc_path = os.path.join(desc_dir, vid, vid+'.json')
        if item['sample_id'] not in processed_id and os.path.exists(desc_path):
            with open(desc_path, 'r') as f2:
                desc_list = json.load(f2)
            for desc in desc_list:
                if desc['start'] < answer_start or desc['start'] > answer_end:
                    tmp = {'sample_id': item['sample_id'], 'vid': vid, 'question': question}
                    sys_input = relevance_scoring_prompt['sys_input']
                    user_input = relevance_scoring_prompt['user_input']
                    user_input = user_input.replace('<question>', question)
                    user_input = user_input.replace('<description>', desc['desc'])
                    score = model_inference_api(sys_input, user_input, args.llm_model, args.api_key)
                    tmp['desc'] = desc['desc']
                    tmp['score'] = int(score)
                    question_desc_score_list.append(tmp)
        if i % 100 == 0:
            with open(score_path, 'w') as f3:
                json.dump(question_desc_score_list, f3)
    with open(score_path, 'w') as f4:
        json.dump(question_desc_score_list, f4)
    return question_desc_score_list


def discriminator_data_process(data_path, desc_dir, score_path):
    input_data = []
    with open(data_path, 'r') as f1:
        data = json.load(f1)
    # with open(score_path, 'r') as f2:
    #     score_data = json.load(f2)
    for item in data:
        vid = item['video_id']
        question = item['question']
        answer_start = item['answer_start_second']
        answer_end = item['answer_end_second']
        desc_path = os.path.join(desc_dir, vid, vid+'.json')
        if os.path.exists(desc_path):
            with open(desc_path, 'r') as f2:
                desc_list = json.load(f2)
            for desc in desc_list:
                if desc['start'] >= answer_start and desc['start'] <= answer_end:
                    input_data.append(((question, desc['desc'], desc['start'], desc['duration']), 1))
                else:
                    input_data.append(((question, desc['desc'], desc['start'], desc['duration']), 0))
                    # for score_item in score_data:
                    #     if score_item['question'] == question and score_item['desc'] == desc['desc']:
                    #         if score_item['score'] >= 4:
                    #             input_data.append(((question, desc['desc'], desc['start'], desc['duration']), 1))
                    #         else:
                    #             input_data.append(((question, desc['desc'], desc['start'], desc['duration']), 0))
    return input_data # list of ((question, desc, start, duration), label)

def dialogue_generate(data_path, subtitle_dir, dialogue_path, args, lang='en'):
    dialogue_list = []
    processed_id = []
    if os.path.exists(dialogue_path):
        with open(dialogue_path, 'r', encoding='utf-8') as f:
            dialogue_list = json.load(f)
            processed_id = [item['sample_id'] for item in dialogue_list]
    print('Already processed ' + str(len(processed_id)))
    with open(data_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
    for i in tqdm(range(len(data))):
        item = data[i]
        vid = item['video_id']
        subtitle_path = os.path.join(subtitle_dir, vid, vid+'.json')
        if item['sample_id'] not in processed_id and os.path.exists(subtitle_path):
            history = {'sample_id': item['sample_id'], 'vid': vid, 'question': item['question'], 'dialogue': []}
            init_question = item['question']
            answer_start = item['answer_start_second']
            answer_end = item['answer_end_second']
            history['answer_start_second'] = answer_start
            history['answer_end_second'] = answer_end
            truth_desc_list = []
            with open(subtitle_path, 'r', encoding='utf-8') as f2:
                desc_list = json.load(f2)
            for desc in desc_list:
                if desc['start'] >= answer_start and desc['start'] <= answer_end:
                    truth_desc_list.append(desc['text'])
            for r in range(args.dialogue_round):
                if lang == 'en':
                    sys_input = further_questioning_prompt['sys_input']
                    user_input = further_questioning_prompt['user_input']
                elif lang == 'zh':
                    sys_input = further_questioning_prompt_zh['sys_input']
                    user_input = further_questioning_prompt_zh['user_input']
                user_input = user_input.replace('<init_question>', init_question)
                user_input = user_input.replace('<hist_dialogue>', str(history['dialogue']))
                user_input = user_input.replace('<description_spans>', str(truth_desc_list))
                next_question = model_inference_api(sys_input, user_input, args.llm_name, args.api_key)
                if lang == 'en':
                    sys_input = yes_no_answering_prompt_2['sys_input']
                    user_input = yes_no_answering_prompt_2['user_input']
                elif lang == 'zh':
                    sys_input = yes_no_answering_prompt_2_zh['sys_input']
                    user_input = yes_no_answering_prompt_2_zh['user_input']
                user_input = user_input.replace('<question>', next_question)
                user_input = user_input.replace('<init_question>', init_question)
                next_answer = model_inference_api(sys_input, user_input, args.llm_name, args.api_key)
                history['dialogue'].append((next_question, next_answer))
            dialogue_list.append(history)
        if i % 100 == 0:
            with open(dialogue_path, 'w', encoding='utf-8') as f3:
                json.dump(dialogue_list, f3, ensure_ascii=False)
    with open(dialogue_path, 'w', encoding='utf-8') as f4:
        json.dump(dialogue_list, f4, ensure_ascii=False)
    return dialogue_list

def dialogue_to_explain(dialogue_path, dialogue_desc_path, args, lang='en'):
    desc_list = []
    processed_id = []
    if os.path.exists(dialogue_desc_path):
        with open(dialogue_desc_path, 'r', encoding='utf-8') as f:
            processed_list = json.load(f)
            processed_id = [item['sample_id'] for item in processed_list]
    print('Already processed ' + str(len(processed_id)))
    with open(dialogue_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    for i in tqdm(range(len(data_list))):
        tmp = data_list[i]
        if tmp['sample_id'] not in processed_id:
            if lang == 'en':
                sys_input = dialogue_summary_prompt['sys_input']
                user_input = dialogue_summary_prompt['user_input']
            elif lang == 'zh':
                sys_input = dialogue_summary_prompt_zh['sys_input']
                user_input = dialogue_summary_prompt_zh['user_input']
            user_input = user_input.replace('<question>', tmp['question'])
            dialogue = tmp['dialogue']
            dialogue_list = ['Inquirer: ' + dial[0] + "User: " + dial[1] for dial in dialogue]
            dialogue_str = '\n'.join(dialogue_list)
            user_input = user_input.replace('<dialogue>', dialogue_str)
            desc = model_inference_api(sys_input, user_input, args.llm_name, args.api_key)
            tmp['description'] = desc
            desc_list.append(tmp)
        if i % 100 == 0:
            with open(dialogue_desc_path, 'w', encoding='utf-8') as f:
                json.dump(desc_list, f, ensure_ascii=False)
    with open(dialogue_desc_path, 'w', encoding='utf-8') as f:
        json.dump(desc_list, f, ensure_ascii=False)
    return desc_list

def selector_data_process(desc_dir, dialogue_path):
    input_data = []
    with open(dialogue_path, 'r') as f1:
        data = json.load(f1)
    for item in data:
        vid = item['vid']
        question = item['question'] # initial question
        dialogue = item['dialogue'] # list of (question, answer) pairs
        dialogue_list = []
        for dialogue_round in dialogue:
            dialogue_list.append(dialogue_round[0] + '|' + dialogue_round[1])
        dialogue_str = '||'.join(dialogue_list)
        answer_start = item['answer_start_second']
        answer_end = item['answer_end_second']
        desc_path = os.path.join(desc_dir, vid, vid+'.json')
        if os.path.exists(desc_path):
            with open(desc_path, 'r') as f2:
                desc_list = json.load(f2)
            for desc in desc_list:
                if desc['start'] >= answer_start and desc['start'] <= answer_end:
                    input_data.append(((desc['desc'], dialogue_str), 1))
                else:
                    input_data.append(((desc['desc'], dialogue_str), 0))
    return input_data # list of ((desc, dialogue), label)


def input_data_process_desc(desc_dir, dialogue_path, ratio=1.0):
    input_data = []
    with open(dialogue_path, 'r') as f1:
        data = json.load(f1)
    max_len = round(len(data) * ratio)
    for item in data[:max_len]:
        vid = item['vid']
        question = item['question'] # initial question
        dialogue = item['dialogue'] # list of (question, answer) pairs
        dialogue_list = []
        for dialogue_round in dialogue:
            dialogue_list.append(dialogue_round[0] + '|' + dialogue_round[1])
        dialogue_str = '||'.join(dialogue_list)
        answer_start = item['answer_start_second']
        answer_end = item['answer_end_second']
        desc_path = os.path.join(desc_dir, vid, vid+'.json')
        if os.path.exists(desc_path):
            with open(desc_path, 'r') as f2:
                desc_list = json.load(f2)
            for i in range(len(desc_list)):
                tmp = desc_list[i]['desc'].split('\nFrame image description: ')
                subtitle = tmp[0].replace('Frame subtitle: ', '')
                if len(tmp) > 1:
                    caption = tmp[1]
                else:
                    caption = 'no caption'
                start = desc_list[i]['start']
                duration = desc_list[i]['duration']
                # 0 for negative, 1 for start positive, 2 for middle positive, 3 for end positive
                if i == 0:
                    if desc_list[i]['start'] < answer_start:
                        input_data.append(((question, dialogue_str, subtitle, caption, start, duration), (0, 0)))
                    elif desc_list[i]['start'] >= answer_start:
                        input_data.append(((question, dialogue_str, subtitle, caption, start, duration), (1, 1)))
                elif i > 0 and i < (len(desc_list) - 1):
                    if desc_list[i]['start'] < answer_start:
                        input_data.append(((question, dialogue_str, subtitle, caption, start, duration), (0, 0)))
                    elif desc_list[i-1]['start'] < answer_start and desc_list[i]['start'] >= answer_start:
                        input_data.append(((question, dialogue_str, subtitle, caption, start, duration), (1, 1)))
                    elif desc_list[i-1]['start'] >= answer_start and desc_list[i]['start'] >= answer_start and desc_list[i+1]['start'] <= answer_end:
                        input_data.append(((question, dialogue_str, subtitle, caption, start, duration), (1, 2))) # 1, 2 for fine grained label
                    elif desc_list[i]['start'] <= answer_end and desc_list[i+1]['start'] > answer_end:
                        input_data.append(((question, dialogue_str, subtitle, caption, start, duration), (1, 3)))
                    elif desc_list[i]['start'] > answer_end:
                        input_data.append(((question, dialogue_str, subtitle, caption, start, duration), (0, 0)))
                else:
                    if desc_list[i]['start'] <= answer_end:
                        input_data.append(((question, dialogue_str, subtitle, caption, start, duration), (1, 3)))
                    elif desc_list[i]['start'] > answer_end:
                        input_data.append(((question, dialogue_str, subtitle, caption, start, duration), (0, 0)))

    return input_data # list of ((question, dialogue_str, subtitle, caption), (coarse_label, fine_label))


class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path):
        self.emb_model = SentenceTransformer(model_name_or_path=model_path)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.emb_model.encode(input).tolist()
        return embeddings

class LocalEmbeddingFunctionBERT(EmbeddingFunction):
    def __init__(self, model_path):
        from transformers import AutoModel, AutoTokenizer

        self._torch = importlib.import_module("torch")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path)

    @staticmethod
    def _normalize(vector: npt.NDArray) -> npt.NDArray:
        """Normalizes a vector to unit length using L2 norm."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def __call__(self, input: Documents) -> Embeddings:
        inputs = self._tokenizer(input, padding=True, truncation=True, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        return [e.tolist() for e in self._normalize(embeddings)]

def index_paper(data_path, subtitle_path, embedding_model_path, window_size=1):
    video_id_list = os.listdir(subtitle_path)
    if '.DS_Store' in video_id_list:
        video_id_list.remove('.DS_Store')
    embedding_path = data_path + '/embedding_custom_context_'+str(window_size)
    exist_store_list = []
    if os.path.exists(embedding_path):
        exist_store_list = os.listdir(embedding_path)
    else:
        os.mkdir(embedding_path)
    for video_id in video_id_list:
        emb_func = LocalEmbeddingFunctionBERT(embedding_model_path)
        emb_store_name = 'emb_'+video_id
        if emb_store_name not in exist_store_list:
            print("Start " + video_id)
            client = chromadb.PersistentClient(path=embedding_path + '/' + emb_store_name)
            collection = client.get_or_create_collection(name=emb_store_name, embedding_function=emb_func)
            subtitle_file = os.path.join(subtitle_path, video_id, video_id+'.json')
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                subtitle_list = json.load(f)
                subtitles = []
                for i in range(len(subtitle_list)):
                    if i < window_size:
                        subtitles.append(' '.join([item['text'] for item in subtitle_list[0 : i+window_size]]))
                    else:
                        subtitles.append(' '.join([item['text'] for item in subtitle_list[i-window_size : i+window_size]]))
                documents = []
                ids = []
                metadatas = []
                count = 0
                if len(subtitles) > 0:
                    for subtitle in subtitles:
                        ids.append(str(count))
                        documents.append(subtitle)
                        metadatas.append({'id': str(count)})
                        count += 1
                    collection.add(documents=documents, ids=ids, metadatas=metadatas)
                else:
                    print(video_id)
                    continue
            print("Finish generating embeddings for "+video_id)
        else:
            continue

def retrieve_context(data_path, video_id, subtitle, question, dialogue_str, embedding_model_path, top_k=20, window_size=1):
    # emb_func = LocalEmbeddingFunction(embedding_model_path)
    emb_func = LocalEmbeddingFunctionBERT(embedding_model_path)
    embedding_path = data_path + '/embedding_custom_context_'+str(window_size)
    store_name = 'emb_'+video_id
    client = chromadb.PersistentClient(path=embedding_path + '/' + store_name)
    collection = client.get_or_create_collection(name=store_name, embedding_function=emb_func)
    if top_k == 0:
        top_k = 1
    res = collection.query(query_texts=[subtitle], n_results=top_k)
    # res = collection.query(query_texts=[question + ' ' + subtitle], n_results=top_k)
    # res = collection.query(query_texts=[question], n_results=top_k)
    # res = collection.query(query_texts=[question + ' ' + dialogue_str], n_results=top_k)
    context_list = res['documents'][0]
    context = [item.replace('\n', ' ') for item in context_list]
    return context 

def get_processed_id(data_file):
    id_dict = {}
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        data = eval(line.strip())
        id_dict[data[0][4]] = 1
    return id_dict

def input_data_process(
        data_path,
        dialogue_path, 
        subtitle_dir, 
        feature_dir, 
        ratio=1.0, 
        augmented_data_path=None, 
        embedding_model_path=None, 
        subtitle_context=1, 
        subtitle_top_p=0.4, 
        use_rewrite=True,
        use_rag=True,
        lang='en'
    ):
    input_data = []
    processed_id_dict = {}
    if augmented_data_path:
        if os.path.exists(augmented_data_path):
            processed_id_dict = get_processed_id(augmented_data_path)
    
    rewriter = None
    if use_rewrite:
        rewriter = SubRewrite(lang=lang)
    stored_rewrite_list = {}

    with open(dialogue_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)

    max_len = round(len(data) * ratio)
    for item in data[:max_len]:
        vid = item['vid']
        # print("Processing video " + vid)
        question = item['question'] # initial question
        dialogue = item['dialogue'] # list of (question, answer) pairs
        dialogue_list = []
        for dialogue_round in dialogue:
            dialogue_list.append(dialogue_round[0] + '|' + dialogue_round[1])
        dialogue_str = '||'.join(dialogue_list)
        answer_start = item['answer_start_second']
        answer_end = item['answer_end_second']
        feature_path = os.path.join(feature_dir, vid)
        subtitle_path = os.path.join(subtitle_dir, vid, vid+'.json')
        if os.path.exists(feature_path):
            feature_file_list = os.listdir(feature_path)
            if '.DS_Store' in feature_file_list:
                feature_file_list.remove('.DS_Store')
            feature_file_list.sort(key=lambda x : int(x.split(vid+'_')[1].split('.pt')[0]))
            with open(subtitle_path, 'r', encoding='utf-8') as f2:
                subtitle_list = json.load(f2)
                full_subtitle_text = ' '.join([item['text'] for item in subtitle_list])
            # ======================
            # subtitle_start_idx = None
            # subtitle_end_idx = None
            # for i in range(len(subtitle_list)):
                # start = subtitle_list[i]['start']
                # duration = subtitle_list[i]['duration']
                # if start >= answer_start:
                    # subtitle_start_idx = i
                    # break
            # for i in range(len(subtitle_list)):
                # start = subtitle_list[i]['start']
                # duration = subtitle_list[i]['duration']
                # if start + duration > answer_end:
                    # subtitle_end_idx = i
                    # break
            # =====================
            for i in range(len(subtitle_list)):
                if i >= len(feature_file_list):
                    break
                if feature_file_list[i] in processed_id_dict:
                    continue
                subtitle_context_list = []
                start = subtitle_list[i]['start']
                duration = subtitle_list[i]['duration']
                current_sample = None
                
                if i == 0:
                    subtitle_context_list = [item['text'] for item in subtitle_list[0 : subtitle_context+1]]
                    subtitle = ' '.join(subtitle_context_list)
                    
                    sub_desc = ''
                    if use_rewrite:
                        if feature_file_list[i] in stored_rewrite_list:
                            sub_desc = stored_rewrite_list[feature_file_list[i]]
                        else:
                            if lang == 'en':
                                rewriter.messages[0]["content"] = subtitle_rewrite_prompt.replace('<all_subtitles>', full_subtitle_text).replace('<subtitle>', subtitle)
                            elif lang == 'zh':
                                rewriter.messages[0]["content"] = subtitle_rewrite_prompt_zh.replace('<all_subtitles>', full_subtitle_text).replace('<subtitle>', subtitle)
                            sub_desc = rewriter.generate()['content']
                            stored_rewrite_list[feature_file_list[i]] = sub_desc

                    context = ''
                    if use_rag:
                        top_k = round(len(subtitle_list) * subtitle_top_p)
                        retrieval_res = retrieve_context(data_path, vid, subtitle, question, dialogue_str, embedding_model_path, top_k=top_k)
                        context = ' | '.join(retrieval_res)
                        subtitle = subtitle.replace('\n', ' ') + ' ||  ' + context
                        # subtitle = context

                    if start < answer_start:
                        current_sample = ((question, dialogue_str, subtitle, sub_desc, feature_file_list[i], start, duration), (0, 0))
                    elif start >= answer_start:
                        # ==================
                        # subtitle_text_list = [item['text'] for item in subtitle_list[subtitle_start_idx: subtitle_end_idx]]
                        # subtitle = ' '.join(subtitle_text_list)
                        # ==================
                        current_sample = ((question, dialogue_str, subtitle, sub_desc, feature_file_list[i], start, duration), (1, 1))
                
                elif i > 0 and i < (len(subtitle_list) - 1):
                    if i-subtitle_context < 0:
                        subtitle_context_list = [item['text'] for item in subtitle_list[0 : i+subtitle_context+1]]
                    elif i+subtitle_context > len(subtitle_list) - 1:
                        subtitle_context_list = [item['text'] for item in subtitle_list[i-subtitle_context : len(subtitle_list)-1]]
                    else:
                        subtitle_context_list = [item['text'] for item in subtitle_list[i-subtitle_context : i+subtitle_context+1]]
                    subtitle = ' '.join(subtitle_context_list)
                    
                    sub_desc = ''
                    if use_rewrite:
                        if feature_file_list[i] in stored_rewrite_list:
                            sub_desc = stored_rewrite_list[feature_file_list[i]]
                        else:
                            if lang == 'en':
                                rewriter.messages[0]["content"] = subtitle_rewrite_prompt.replace('<all_subtitles>', full_subtitle_text).replace('<subtitle>', subtitle)
                            elif lang == 'zh':
                                rewriter.messages[0]["content"] = subtitle_rewrite_prompt_zh.replace('<all_subtitles>', full_subtitle_text).replace('<subtitle>', subtitle)
                            sub_desc = rewriter.generate()['content']
                            stored_rewrite_list[feature_file_list[i]] = sub_desc

                    context = ''
                    if use_rag:
                        top_k = round(len(subtitle_list) * subtitle_top_p)
                        retrieval_res = retrieve_context(data_path, vid, subtitle, question, dialogue_str, embedding_model_path, top_k=top_k)
                        context = ' | '.join(retrieval_res)
                        subtitle = subtitle.replace('\n', ' ') + ' || ' + context
                        # subtitle = context

                    if start < answer_start:
                        current_sample = ((question, dialogue_str, subtitle, sub_desc, feature_file_list[i], start, duration), (0, 0))
                    elif subtitle_list[i-1]['start'] < answer_start and start >= answer_start:
                        # ==================
                        # subtitle_text_list = [item['text'] for item in subtitle_list[subtitle_start_idx: subtitle_end_idx]]
                        # subtitle = ' '.join(subtitle_text_list)
                        # ==================
                        current_sample = ((question, dialogue_str, subtitle, sub_desc, feature_file_list[i], start, duration), (1, 1))
                    elif subtitle_list[i-1]['start'] >= answer_start and start >= answer_start and subtitle_list[i+1]['start'] <= answer_end:
                        # ==================
                        # subtitle_text_list = [item['text'] for item in subtitle_list[subtitle_start_idx: subtitle_end_idx]]
                        # subtitle = ' '.join(subtitle_text_list)
                        # ==================
                        current_sample = ((question, dialogue_str, subtitle, sub_desc, feature_file_list[i], start, duration), (1, 1))
                    elif start <= answer_end and subtitle_list[i+1]['start'] > answer_end:
                        # ==================
                        # subtitle_text_list = [item['text'] for item in subtitle_list[subtitle_start_idx: subtitle_end_idx]]
                        # subtitle = ' '.join(subtitle_text_list)
                        # ==================
                        current_sample = ((question, dialogue_str, subtitle, sub_desc, feature_file_list[i], start, duration), (1, 1))
                    elif start > answer_end:
                        current_sample = ((question, dialogue_str, subtitle, sub_desc, feature_file_list[i], start, duration), (0, 2))
                
                else:
                    subtitle_context_list = [item['text'] for item in subtitle_list[i-subtitle_context : len(subtitle_list)-1]]
                    subtitle = ' '.join(subtitle_context_list)

                    sub_desc = ''
                    if use_rewrite:
                        if feature_file_list[i] in stored_rewrite_list:
                            sub_desc = stored_rewrite_list[feature_file_list[i]]
                        else:
                            if lang == 'en':
                                rewriter.messages[0]["content"] = subtitle_rewrite_prompt.replace('<all_subtitles>', full_subtitle_text).replace('<subtitle>', subtitle)
                            elif lang == 'zh':
                                rewriter.messages[0]["content"] = subtitle_rewrite_prompt_zh.replace('<all_subtitles>', full_subtitle_text).replace('<subtitle>', subtitle)
                            sub_desc = rewriter.generate()['content']
                            stored_rewrite_list[feature_file_list[i]] = sub_desc

                    context = ''
                    if use_rag:
                        top_k = round(len(subtitle_list) * subtitle_top_p)
                        retrieval_res = retrieve_context(data_path, vid, subtitle, question, dialogue_str, embedding_model_path, top_k=top_k)
                        context = ' | '.join(retrieval_res)
                        subtitle = subtitle.replace('\n', ' ') + ' || ' + context
                        # subtitle = context

                    if start <= answer_end:
                        # ==================
                        # subtitle_text_list = [item['text'] for item in subtitle_list[subtitle_start_idx: subtitle_end_idx]]
                        # subtitle = ' '.join(subtitle_text_list)
                        # ==================
                        current_sample = ((question, dialogue_str, subtitle, sub_desc, feature_file_list[i], start, duration), (1, 1))
                    elif start > answer_end:
                        current_sample = ((question, dialogue_str, subtitle, sub_desc, feature_file_list[i], start, duration), (0, 2))
            
                input_data.append(current_sample)
                if use_rag:
                    with open(augmented_data_path, 'a', encoding='utf-8') as f:
                        f.write(str(current_sample) + '\n')

    return input_data # list of ((question, dialogue_str, subtitle, sub_desc, feature_id, start, duration), (coarse_label, fine_label))

def load_data(dataset_dir, dataset_name, mode, data_ratio=1.0, subtitle_ratio=0.8, use_dialogue=True, dialogue_round=3, use_rag=True, use_rewrite_dialogue=True, use_rewrite_subtitle=True):
    input_data = []
    with open(os.path.join(dataset_dir, dataset_name, mode+'_dialogue_desc.json'), 'r', encoding='utf-8') as f:
        dialogue_desc = json.load(f)
    data_file = os.path.join(dataset_dir, dataset_name, mode+'_custom_rag_augmented.jsonl')
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_num = round(len(lines) * data_ratio)
    for line in lines[:data_num]:
        tmp = eval(line.strip())
        sample = tmp[0]
        label = tmp[1]
        if len(sample[2]) == 0:
            continue
        # subtitle_select = sample[2].split('||')[0] + '||' + '|'.join(sample[2].split('||')[1].split('|')[:subtitle_top_k])
        subtitle = sample[2].split('||')[0]
        context_list = sample[2].split('||')[1].split('|')
        # context_list = sample[2].split('|')
        context_num = len(context_list)
        select_context_num = max(round(context_num * subtitle_ratio), 15)
        select_context = '|'.join(context_list[:select_context_num])
        # subtitle_select = sample[2].split('||')[0] + '||' + select_context
        subtitle_desc = sample[3]
        dialogue_select = '||'.join(sample[1].split('||')[:dialogue_round])

        question_desc = dialogue_select
        vid = sample[4].replace('_' + sample[4].split('_')[-1], '')
        for item in dialogue_desc:
            if item['vid'] == vid:
                question_desc = item['description']
        dialogue_select = '||'.join(sample[1].split('||')[:dialogue_round])
        # conversation rewrite + retrival + subtitle rewrite
        if use_dialogue and use_rag and use_rewrite_dialogue and use_rewrite_subtitle:
            sample_select = ((sample[0], question_desc, subtitle_desc, select_context, sample[4], sample[5], sample[6]), label)
        # conversation rewrite + subtitle rewrite
        elif use_dialogue and use_rewrite_dialogue and use_rewrite_subtitle and not use_rag:
            sample_select = ((sample[0], question_desc, subtitle_desc, '', sample[4], sample[5], sample[6]), label)
        # conversation + retrieval + subtitle rewrite
        elif use_dialogue and use_rag and use_rewrite_subtitle and not use_rewrite_dialogue:
            sample_select = ((sample[0], dialogue_select, subtitle_desc, select_context, sample[4], sample[5], sample[6]), label)
        # retrival + subtitle rewrite
        elif use_rag and use_rewrite_subtitle and not use_dialogue:
            sample_select = ((sample[0], '', subtitle_desc, select_context, sample[4], sample[5], sample[6]), label)
        # conversation rewrite + retrieval
        elif use_dialogue and use_rag and use_rewrite_dialogue and not use_rewrite_subtitle:
            sample_select = ((sample[0], question_desc, subtitle, select_context, sample[4], sample[5], sample[6]), label)
        # no strategy
        elif not use_dialogue and not use_rag and not use_rewrite_subtitle:
            sample_select = ((sample[0], '', subtitle, '', sample[4], sample[5], sample[6]), label)
        else:
            raise ValueError("Unsupported setting values")
        input_data.append(sample_select)
    return input_data

def subtitle_relevancy_train_with_context(data_path, dialogue_path, subtitle_dir, mode='train', context=1):
    # construct training set for text pair classification BERT
    # with each question+subtitle and its context
    # where two subtitles both inside and outside the answer span with label 1
    # and one within and the other without the answer span with label 0
    samples = []
    pos_samples = []
    neg_samples = []
    with open(dialogue_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        data = data[:round(len(data)*1.0)]

    if mode == 'train':
        with open(dialogue_path.replace('train', 'test'), 'r', encoding='utf-8') as f2:
            data_test = json.load(f2)
        data.extend(data_test)

    for item in data:
        vid = item['vid']
        question = item['question']
        print("Processing video " + vid)
        answer_start = item['answer_start_second']
        answer_end = item['answer_end_second']

        subtitle_file = os.path.join(subtitle_dir, vid, vid+'.json')
        if os.path.exists(subtitle_file):
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                subtitle_list = json.load(f)
            in_answer_subs = []
            before_answer_subs = []
            after_answer_subs = []
            for i in range(len(subtitle_list)):
                if i < context:
                    sub_text = ' '.join([item['text'] for item in subtitle_list[0 : i+context]])
                else:
                    sub_text = ' '.join([item['text'] for item in subtitle_list[i-context : i+context]])
                sub_text = sub_text.replace('\n', '\\n')

                sub_start = subtitle_list[i]['start']
                sub_end = subtitle_list[i]['start']+subtitle_list[i]['duration']
                if sub_start >= answer_start and sub_end <= answer_end:
                    in_answer_subs.append(sub_text)
                elif sub_start < answer_start:
                    before_answer_subs.append(sub_text)
                elif sub_end > answer_end:
                    after_answer_subs.append(sub_text)

            if len(in_answer_subs) >= 2 and len(before_answer_subs) >= 2 and len(after_answer_subs) >= 2:
                sample_num = round(len(in_answer_subs) / 2)
                sample_num = max([sample_num, 3])
                for i in range(sample_num):
                    pos_pair = random.sample(in_answer_subs, 2)
                    pos_samples.append([question + ' ' + pos_pair[0], pos_pair[1], 1])
                    pos_pair = random.sample(in_answer_subs, 2)
                    pos_samples.append([question + ' ' + pos_pair[0], pos_pair[1], 1])
                    pos_pair = random.sample(before_answer_subs, 2)
                    pos_samples.append([question + ' ' + pos_pair[0], pos_pair[1], 1])
                    pos_pair = random.sample(after_answer_subs, 2)
                    pos_samples.append([question + ' ' + pos_pair[0], pos_pair[1], 1])
                    neg_pair = random.sample(in_answer_subs, 1) + random.sample(before_answer_subs, 1)
                    neg_samples.append([question + ' ' + neg_pair[0], neg_pair[1], 0])
                    neg_pair = random.sample(before_answer_subs, 1) + random.sample(in_answer_subs, 1)
                    neg_samples.append([question + ' ' + neg_pair[0], neg_pair[1], 0])
                    neg_pair = random.sample(in_answer_subs, 1) + random.sample(after_answer_subs, 1)
                    neg_samples.append([question + ' ' + neg_pair[0], neg_pair[1], 0])
                    neg_pair = random.sample(after_answer_subs, 1) + random.sample(in_answer_subs, 1)
                    neg_samples.append([question + ' ' + neg_pair[0], neg_pair[1], 0])
    
    samples.extend(pos_samples)
    samples.extend(neg_samples)
    random.shuffle(samples)
    with open(data_path + '/' + mode + '_rel.csv','w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text1', 'text2', 'label'])
        writer.writerows(samples)

if __name__ == '__main__':
    args = set_argument()
    dataset = 'cmivqa'
    data_dir = '../dataset'
    data_path = os.path.join(data_dir, dataset)
    subtitle_dir = os.path.join(data_dir, dataset, 'video_process', 'subtitles')
    # dialogue_path_raw = os.path.join(data_dir, dataset, 'train_dialogue.json')
    dialogue_path = os.path.join(data_dir, dataset, 'test_dialogue_desc.json')
    # dialogue_generate(os.path.join(data_path, 'test.json'), subtitle_dir, dialogue_path_raw, args, lang='zh')
    # dialogue_to_explain(dialogue_path_raw, dialogue_path, args, lang='zh')
    # pdb.set_trace()
    embedding_model_path = os.path.join(data_dir, dataset, 'subtitle_sim_bert')
    
    # embedding_model_path='/data/users/bitlab/models/bert-base-uncased'
    # index_paper(data_path, subtitle_dir, embedding_model_path)
    # pdb.set_trace()
    
    
    # subtitle_relevancy_train_with_context(data_path, dialogue_path, subtitle_dir, mode='train', context=1)
    # pdb.set_trace()

    feature_dir = os.path.join(data_dir, dataset, 'video_process', 'framefeatures')
    augmented_data_path = os.path.join(data_dir, dataset, 'test_custom_rag_augmented.jsonl')
    input_data_process(
            data_path,
            dialogue_path, 
            subtitle_dir, 
            feature_dir, 
            augmented_data_path=augmented_data_path, 
            embedding_model_path=embedding_model_path, 
            ratio = 1.0,
            subtitle_context=1, 
            subtitle_top_p=0.1, 
            use_rag=True,
            lang='zh'
        )
    
