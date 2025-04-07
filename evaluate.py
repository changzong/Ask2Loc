import torch
import json
import math
import numpy as np
import pdb

def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    if (union[1] - union[0]) == 0:
        return 0.0
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)

def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0

def evaluate(model, test_loader, test_data_path, batch_size, device):
    # device = torch.device("cpu")
    ious = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    model.eval()
    query_list = tuple()
    clip_feature_id_list = tuple()
    input_ids_q_list = torch.tensor([]).to(device)
    input_ids_qd_list = torch.tensor([]).to(device)
    input_ids_sub_list = torch.tensor([]).to(device)
    input_ids_con_list = torch.tensor([]).to(device)
    labels_rel_list = torch.tensor([]).to(device)
    labels_evt_list = torch.tensor([]).to(device)
    start_list = torch.tensor([]).to(device)
    duration_list = torch.tensor([]).to(device)
    current_query = ''
    for input_data in test_loader:
        question = input_data["question"]
        clip_feature_id = input_data["clip_feature_id"]
        input_ids_q = input_data["input_ids_question"].to(device)
        input_ids_qd = input_data["input_ids_question_deep"].to(device)
        input_ids_sub = input_data["input_ids_subtitle_desc"].to(device)
        input_ids_con = input_data["input_ids_subtitle_context"].to(device)
        start = input_data["start"].to(device)
        duration = input_data["duration"].to(device)
        labels_rel = input_data["labels_relevancy"].to(device)
        labels_evt = input_data["labels_event"].to(device)
        if query_list:
            if question[0] == current_query:
                query_list = query_list + question
                clip_feature_id_list = clip_feature_id_list + clip_feature_id
                input_ids_q_list = torch.cat((input_ids_q_list, input_ids_q), dim=0)
                input_ids_qd_list = torch.cat((input_ids_qd_list, input_ids_qd), dim=0)
                input_ids_sub_list = torch.cat((input_ids_sub_list, input_ids_sub), dim=0)
                input_ids_con_list = torch.cat((input_ids_con_list, input_ids_con), dim=0)
                labels_rel_list = torch.cat((labels_rel_list, labels_rel), dim=0)
                labels_evt_list = torch.cat((labels_evt_list, labels_evt), dim=0)
                start_list = torch.cat((start_list, start), dim=0)
                duration_list = torch.cat((duration_list, duration), dim=0)
            else:
                time_min, time_max = 0, 0
                answer_span = [0, 0]
                count = math.ceil(input_ids_q_list.shape[0] / batch_size)
                pos_index_list = []
                bou_index_list = []
                idx_start, idx_end = 0, 0
                for i in range(count):
                    idx_start = batch_size * i
                    if i < count - 1:
                        idx_end = batch_size * (i+1)
                    else:
                        idx_end = input_ids_q_list.shape[0]
                    loss, pos_index = model(
                        input_ids_q_list[idx_start: idx_end], 
                        input_ids_qd_list[idx_start: idx_end],
                        input_ids_sub_list[idx_start: idx_end],
                        input_ids_con_list[idx_start: idx_end], 
                        clip_feature_id_list[idx_start: idx_end], 
                        labels_rel_list[idx_start: idx_end], 
                        labels_evt_list[idx_start: idx_end]
                    )
                    tmp1 = [i * batch_size + item for item in pos_index]
                    pos_index_list.extend(tmp1)
                    del loss
                print("positive index: {}".format(str(pos_index_list)))
                start_list  = start_list.tolist()
                duration_list = duration_list.tolist()

                for item in test_data:
                    if query_list[0].split('Question: ')[1] == item["question"]:
                        answer_span = [item['answer_start_second'], item['answer_end_second']]
                        print('Sample: '+str(item['sample_id']))

                        if pos_index_list:
                            start_selected = [start_list[i] for i in pos_index_list]
                            duration_selected = [duration_list[i] for i in pos_index_list]
                            time_min = min(start_selected)
                            time_max = max(start_selected) + duration_selected[start_selected.index(max(start_selected))]
                        predict_pos_span = [time_min, time_max]
                        print('predicted positive span' + str(predict_pos_span))
                        print('answer span: ' + str(answer_span))
                        iou = calculate_iou(predict_pos_span, answer_span)
                        print(str(iou) + '\n')
                        ious.append(iou)
                    else:
                        continue
                query_list = question
                clip_feature_id_list = clip_feature_id
                input_ids_q_list = input_ids_q
                input_ids_qd_list = input_ids_qd
                input_ids_sub_list = input_ids_sub
                input_ids_con_list = input_ids_con
                labels_rel_list = labels_rel
                labels_evt_list = labels_evt
                start_list = start
                duration_list = duration
        else:
            query_list = question
            clip_feature_id_list = clip_feature_id
            input_ids_q_list = input_ids_q
            input_ids_qd_list = input_ids_qd
            input_ids_sub_list = input_ids_sub
            input_ids_con_list = input_ids_con
            labels_rel_list = labels_rel
            labels_evt_list = labels_evt
            start_list = start
            duration_list = duration
        current_query = question[0]
    print("\n")
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0
    score_str = ["Rank@1, IoU=0.3: {:.2f}".format(r1i3)]
    score_str += ["Rank@1, IoU=0.5: {:.2f}".format(r1i5)]
    score_str += ["Rank@1, IoU=0.7: {:.2f}".format(r1i7)]
    score_str += ["mean IoU: {:.2f}".format(mi)]
    print(score_str)
