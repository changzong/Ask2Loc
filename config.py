import argparse

def csv_list(string):
    return string.split(',')

def set_argument():
    parser = argparse.ArgumentParser(prog='RCVTA')
    parser.add_argument('--project', type=str, default='RCVTA')
    # data load settings
    parser.add_argument('--dataset_name', type=str, default='cmivqa')
    parser.add_argument('--dataset_dir', type=str, default='../dataset')
    parser.add_argument('--use_dialogue', action='store_true', default=True)
    parser.add_argument('--dialogue_round', type=int, default=3)
    parser.add_argument('--use_rag', action='store_true', default=True)
    parser.add_argument('--use_rewrite_dialogue', action='store_true', default=True)
    parser.add_argument('--use_rewrite_subtitle', action='store_true', default=True)
    parser.add_argument('--subtitle_folder', type=str, default='subtitles')
    parser.add_argument('--subtitle_context', type=int, default=3)
    parser.add_argument('--subtitle_ratio', type=float, default=1.0)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    # training feature settings
    parser.add_argument('--use_global_subtitle', action='store_true', default=False)
    parser.add_argument('--use_vision', action='store_true', default=False)
    parser.add_argument('--clip_feature_folder', type=str, default='framefeatures')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--video_feature_dim', type=int, default=1024) # 1408
    parser.add_argument('--hidden_dim', type=int, default=1024) # 768
    parser.add_argument('--lm_name', type=str, default='../../models/deberta-v2-chinese')
    # training settings
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--cuda_name', type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument("--warmup_epochs", type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--grad_steps", type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--patience", type=float, default=2)

    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini')
    parser.add_argument('--api_key', type=str, default='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')

    args = parser.parse_args()
    return args
