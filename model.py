import torch
import torch.nn as nn
from feature_fusion import FeatureFusion
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pdb

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class RCVTA(nn.Module):
    def __init__(self, args, device):
        super(RCVTA, self).__init__()
        if args.use_vision:
            self.fusion_model = FeatureFusion(args, device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(args.lm_name, num_labels=2).to(device)
        self.word_embedding = self.detection_model.get_input_embeddings().to(device)
        if args.use_global_subtitle:
            self.cls_projector = nn.Linear(args.hidden_dim, args.hidden_dim).to(device)
            nn.init.xavier_normal_(self.cls_projector.weight)
        self.args = args
        self.device = device

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param

    def get_subtitle_feature(self, clip_feature_ids):
        batch_feature_list = []
        batch_attention_mask = []
        for fid in clip_feature_ids:
            tail = fid.split('_')[-1]
            vid = fid.replace('_'+tail, '')
            subtitle_feature_path = os.path.join(self.args.dataset_dir, self.args.dataset_name, 'video_process', 'subtitle_features', vid, vid+'.pt')
            subtitle_feature = torch.load(subtitle_feature_path, map_location=self.args.cuda_name, weights_only=True)
            batch_feature_list.append(subtitle_feature)
            batch_attention_mask.append([1])
        feature_emb = torch.stack(batch_feature_list).to(self.device) # batch_size * 1 * hidden_dim
        attention_mask = torch.tensor(batch_attention_mask).to(self.device) # batch_size * 1
        return feature_emb, attention_mask

    def forward(
        self, 
        input_ids_question,
        input_ids_question_deep,
        input_ids_subtitle_desc, 
        input_ids_subtitle_context, 
        clip_feature_ids,
        labels_rel, 
        labels_evt, 
        alpha=1.0
    ):
        batch_question_emb = None
        question_attention_masks = None
        if self.args.use_dialogue:
            batch_question_emb = self.word_embedding(input_ids_question_deep)
        else:
            batch_question_emb = self.word_embedding(input_ids_question)
        batch_fusion_emb = None
        fusion_attention_masks = None
        if self.args.use_vision:
            if self.args.use_rag:
                fusion_outputs = self.fusion_model(input_ids_subtitle_context, clip_feature_ids)
                batch_fusion_emb = torch.cat([fusion_outputs, self.word_embedding(input_ids_subtitle_context)], dim=1)
            else:
                fusion_outputs = self.fusion_model(input_ids_subtitle_desc, clip_feature_ids)
                batch_fusion_emb = torch.cat([fusion_outputs, self.word_embedding(input_ids_subtitle_desc)], dim=1)
        else:
            if self.args.use_rag:
                fusion_outputs = self.word_embedding(input_ids_subtitle_context)
            else:
                fusion_outputs = self.word_embedding(input_ids_subtitle_desc)
            batch_fusion_emb = fusion_outputs

        batch_size = batch_fusion_emb.shape[0]

        # cls_emb = self.word_embedding(torch.tensor(self.tokenizer.cls_token_id).to(self.device))
        # batch_cls_emb = cls_emb.repeat(batch_size, 1, 1)
        # sep_emb = self.word_embedding(torch.tensor(self.tokenizer.sep_token_id).to(self.device))
        # batch_sep_emb = sep_emb.repeat(batch_size, 1, 1) # batch_size * 1 * hidden_dim

        # Get global subtitle features
        if self.args.use_global_subtitle:
            global_sub_features, g_attention_mask = self.get_subtitle_feature(clip_feature_ids) # batch_size * 1 * hidden_dim
            global_sub_features = self.cls_projector(global_sub_features)
            batch_fusion_emb = torch.cat([global_sub_features, batch_fusion_emb], dim=1)

        combined_embeds = torch.cat([batch_fusion_emb, batch_question_emb], dim=1)
        # combined_embeds = torch.cat([batch_cls_emb, batch_question_emb, batch_sep_emb, batch_fusion_emb], dim=1)
        combined_attention_mask = torch.ones((batch_size, combined_embeds.shape[1]), dtype=torch.long, device=self.device)

        outputs = self.detection_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            return_dict=True,
            labels=labels_rel
        )
        loss = outputs.loss
        logits = outputs.logits
        pred = logits.max(-1).indices
        pos_index = (pred == 1).nonzero(as_tuple=True)[0].tolist()

        return loss, pos_index

