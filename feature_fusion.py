import os
import torch
import pickle
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pdb


class FeatureFusion(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            args.lm_name,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # device_map=args.lm_cuda_name
        ).to(device)
        self.word_embedding = self.model.get_input_embeddings().to(device)
        if args.use_vision:
            self.video_projection = nn.Linear(args.video_feature_dim, args.hidden_dim).to(device)
            self.fusion_token = nn.Parameter(torch.randn(1, 1, args.hidden_dim)).to(device)
        self.args = args
        self.device = device
        if not self.args.use_cuda:
            torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())

    def get_clip_feature(self, clip_feature_ids):
        batch_feature_list = []
        batch_attention_mask = []
        for fid in clip_feature_ids:
            tail = fid.split('_')[-1]
            vid = fid.replace('_'+tail, '')
            clip_feature_path = os.path.join(self.args.dataset_dir, self.args.dataset_name, 'video_process', self.args.clip_feature_folder, vid, fid)
            if fid.split('.')[-1] == 'pkl':
                with open(clip_feature_path, 'rb') as f:
                    clip_feature = pickle.load(f)
                clip_feature = clip_feature.squeeze(0).to(self.device)
            elif fid.split('.')[-1] == 'pt':
                clip_feature = torch.load(clip_feature_path, weights_only=False)
                clip_feature = clip_feature.unsqueeze(0).to(self.device)
            else:
                clip_feature = None
            batch_feature_list.append(clip_feature)
            batch_attention_mask.append([1] * clip_feature.shape[0])

        max_length = max([x.shape[0] for x in batch_feature_list])
        print("max clip length: " + str(max_length))
        current_batch_len = len(batch_feature_list)
        for i in range(current_batch_len):
            pad_length = max_length - batch_feature_list[i].shape[0]
            pad_tensor = torch.zeros(pad_length, batch_feature_list[i].shape[1]).to(self.device)
            batch_feature_list[i] = torch.cat([pad_tensor, batch_feature_list[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        feature_emb = torch.stack(batch_feature_list).to(self.device) # batch_size * seq_len * visual_hidden_dim
        attention_mask = torch.tensor(batch_attention_mask).to(self.device) # batch_size * seq_len
        return feature_emb, attention_mask

    def forward(self, input_ids_subtitle, clip_feature_ids):
        subtitle_outputs = None
        subtitle_visual_fusion_outputs = None
        input_ids_subtitle = input_ids_subtitle.to(self.device)
        combined_mask = None
        subtitle_embeds = self.word_embedding(input_ids_subtitle) # batch_size * seq_len * hidden_dim

        clip_features, clip_attention_mask = self.get_clip_feature(clip_feature_ids)
        clip_embeds = self.video_projection(clip_features) # batch_size * seq_len * hidden_dim
        batch_size, num_clip_tokens, _ = clip_embeds.shape
        fusion_tokens = self.fusion_token.expand(batch_size, -1, -1)
        combined_embeds = torch.cat([fusion_tokens, clip_embeds, subtitle_embeds], dim=1)
        total_seq_len = 1 + num_clip_tokens + subtitle_embeds.shape[1]
        token_type_ids = torch.zeros((batch_size, total_seq_len), dtype=torch.long, device=self.device)
        token_type_ids[:, 1+num_clip_tokens:] = 1
        combined_mask = torch.ones((batch_size, total_seq_len), dtype=torch.long, device=self.device)
        subtitle_visual_fusion_outputs = self.model(inputs_embeds=combined_embeds, token_type_ids=token_type_ids, attention_mask=combined_mask)
       
        fusion_features = subtitle_visual_fusion_outputs.hidden_states[-1][:, 0, :]
        return fusion_features.unsqueeze(1) # batch_size * 1 * hidden_dim


if __name__ == '__main__':
    pass
