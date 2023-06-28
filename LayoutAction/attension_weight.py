from tkinter import W
from tracemalloc import start
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from Visualizer.visualizer  import get_local
get_local.activate()

import os
import argparse
import torch
from dataset import get_dataset
from model import GPT, GPTConfig
from trainer import TrainerConfig
from utils import set_seed
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from utils import sample
import json

def visualize_weight(in_token, out_token, attention_weight, img_path="./test/weight.jpg"):
    '''
    attention_weight = torch.randn(8, 8).numpy()
    token = ["token1", "token2", "token3", "token4", "token5", "token6", "token7", "token8"]
    '''
    # attention weight visualization
    f, ax = plt.subplots(figsize=(20, 16))
    ax.xaxis.tick_top()
    pd_weight = pd.DataFrame(attention_weight, index=out_token, columns=in_token)
    sns.heatmap(pd_weight, linewidths=.5, cmap='YlGnBu', ax=ax)
    f.savefig(img_path)
    plt.close(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--exp", default="attension_visualization", help="experiment name")
    parser.add_argument("--log_dir", default="./output/logs", help="/path/to/logs/dir")
    parser.add_argument("--dataset", choices=["rico", "publaynet", "infoppt"], default="publaynet", const='bbox',nargs='?')
    parser.add_argument("--split", choices=['train', 'val', 'test'], default="test", const='split', nargs='?')
    parser.add_argument("--device", type=int, default=0)

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int)
    
    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.dataset)
    samples_dir = os.path.join(log_dir, args.exp)
    os.makedirs(samples_dir, exist_ok=True)
    
    set_seed(args.seed)

    test_dataset = get_dataset(args.dataset, args.split)

    mconf = GPTConfig(test_dataset.vocab_size, test_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)  # a GPT-1
    model = GPT(mconf)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_path = os.path.join("pretrained_model", f"{args.dataset}.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=f"cuda:{args.device}"))
    else:
        raise ValueError(f"model path:{model_path} dose not exist.")

    loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                        batch_size=args.batch_size,
                        num_workers=0)

    pbar = tqdm(enumerate(loader), total=len(loader))

    # weight_file_name = "00_00_0"
    # weight_dict = torch.load(os.path.join(samples_dir, f"{weight_file_name}.pt"))
    # print(weight_dict[f"{weight_file_name}_19_6"][-1])
    '''
    [2.0022555e-04 1.9328553e-03 1.3431879e-03 6.1349355e-02 8.5515273e-04
    1.9777175e-03 7.5255209e-01 4.3972247e-04 1.5876114e-03 6.3911021e-02
    4.3213821e-04 1.6947236e-03 1.3798564e-02 1.2698360e-02 1.4340589e-03
    2.3499574e-03 4.6844365e-05 6.5156363e-02 1.6239997e-02]  ---->  7.5255209e-01: 证明在生成margin的ref obj时， 考虑到了ref obj的y坐标
        
    other findings:
    print(weight_dict[f"{weight_file_name}_19_7"][-1])   ----> 参考了h的信息，因为marigin的计算也需要ref obj的h信息
    print(weight_dict[f"{weight_file_name}_29_5"][-1])   ----> copy x
    print(weight_dict[f"{weight_file_name}_35_6"][-1])   ----> copy w
    print(weight_dict[f"{weight_file_name}_35_7"][-1])   ----> copy w
    '''

    with torch.no_grad():
        for it, (x, y) in pbar:
            x_cond = x.to(device)
            layouts = sample(model, x_cond[:, :1], steps=test_dataset.max_length,
                            temperature=1.0, sample=True, top_k=5, only_label=True, gt=x_cond).detach().cpu().numpy()
            cache = get_local.cache
            attension_map_list = cache["CausalSelfAttention.forward"]

            for i, out_layout in enumerate(layouts):
                save_result = {}  
                box_and_label = test_dataset.render_normalized_layout(out_layout)
                layout = test_dataset.render(out_layout)
                layout.save(os.path.join(samples_dir, f'{it:02d}_{i:02d}_ours.png'))
                
                # len(attension_map_list) == test_dataset.max_length * config.n_layer

                token = test_dataset.layout2token(out_layout)
                pass_idx = 1

                for token_idx in range(2, len(token)):
                    if (token_idx - 1) % 13 == 0:
                        continue
                    input_token = token[:token_idx]
                    output_token = token[1:token_idx+1]
                    atten_map_idx = args.n_layer * pass_idx - 1
                    cur_atten = attension_map_list[atten_map_idx]
                    pass_idx += 1
                    
                    if input_token[-1] == "EOS":
                        break

                    if input_token[-1] in ["COPY", "SPACE"]:
                        ref_obj_index = int(output_token[-1])
                        all_prev_obj_num = (token_idx-1) // 13
                        ref_obj_idx = all_prev_obj_num - ref_obj_index
                        ref_obj_start = 1 + 13 * ref_obj_idx
                        ref_obj_end = 1 + 13 * (ref_obj_idx + 1)
                        for head_idx in range(args.n_head):
                            max_atten_weight = cur_atten[i, head_idx, -1, :].max()
                            if max_atten_weight in cur_atten[i, head_idx, -1, ref_obj_start:ref_obj_end+1]:
                                save_result[f"{it:02d}_{i:02d}_{token_idx}_{head_idx}"] = cur_atten[i, head_idx, :, :]

                                img_path = os.path.join(samples_dir, f"{it:02d}_{i:02d}_{token_idx}_{head_idx}.png")
                                visualize_weight(input_token, output_token, cur_atten[i, head_idx, :, :], img_path=img_path)
                torch.save(save_result, os.path.join(samples_dir, f"{it:02d}_{i:02d}.pt"))
            cache["CausalSelfAttention.forward"] = []

            # attension_map_list: 
            # [
            #     first layer:[[batch_size, n_head, input_sequence_num, input_sequence_num]],
            #     second layer:[],
            #     ...
            #     sixth layer:[],
            #     first layer:[],
            #     ...
            # ]
                