from ast import arg
from doctest import Example
from email.policy import default
import enum
from genericpath import exists
import os
import argparse
from unittest import result
from matplotlib.pyplot import axis
import torch
from dataset import get_dataset
from model import GPT, GPTConfig
from trainer import TrainerConfig
from utils import set_seed
import pickle
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import numpy as np
from collections import defaultdict 
from functools import partial
from multiprocessing import Pool, Value
import atexit
from utils import sample
import shutil

def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def add_zeros_element(box, max_element_num=9):
    if box.shape[0] != max_element_num:
        add_num = max_element_num - box.shape[0]
        if isinstance(box, np.ndarray):
            box = np.concatenate((box, np.zeros((add_num, 4))), axis=0)
        elif isinstance(box, torch.Tensor):
            box = torch.cat((box, torch.zeros((add_num, 4), device=box.device)), dim=0)
    return box

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--exp", default="case_study", help="experiment name")
    parser.add_argument("--log_dir", default="./output/logs", help="/path/to/logs/dir")
    parser.add_argument("--dataset", choices=["rico", "publaynet", "infoppt"], default="publaynet", const='bbox',nargs='?')
    parser.add_argument("--split", choices=['train', 'val', 'test'], default="test", const='split', nargs='?')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--evaluate_name", type=str, default="generated_layouts.pkl")
    parser.add_argument("--command", choices=["save_platte", "dataset_distribution", "generate_ours_case", "cal_iou"], default="save_platte", const='save_platte',nargs='?')
    parser.add_argument("--generate_type",choices=["category_generate", "completion_generate", "random_generate"], default="category_generate", const='category_generate',nargs='?')
    parser.add_argument("--save_num", type=int, default=10)
    parser.add_argument("--min_box_num", type=int, default=3)
    parser.add_argument("--is_train_iou", action="store_true", default=False)
    parser.add_argument("--select_samples_dir", type=str, default=None)

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


    def save_platte_color():
        from PIL import Image, ImageDraw, ImageOps
        W = 512
        H = 512

        img = Image.new('RGB', (W, H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        pace = 30
        wh = 20
        x1, y1, x2, y2 = [0, 0, 512, wh]
        for i in range(len(test_dataset.colors)):
            # x1, y1, x2, y2 = box
            
            col = test_dataset.colors[i]
            draw.rectangle([x1, y1, x2, y2],
                            outline=tuple(col) + (200,),
                            fill=tuple(col) + (64,),
                            width=2)
            y1 += pace
            y2 += pace
        img = ImageOps.expand(img, border=2)
        img.save(os.path.join(samples_dir, f"platte_{args.dataset}.png"))

    def cal_dataset_distribution():
        loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                            batch_size=args.batch_size,
                            num_workers=0)

        pbar = tqdm(enumerate(loader), total=len(loader))

        train_dataset = get_dataset(args.dataset, "train")
        train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True,
                            batch_size=args.batch_size,
                            num_workers=0)

        train_pbar = tqdm(enumerate(train_loader), total=len(loader))

        valid_dataset = get_dataset(args.dataset, "val")
        val_loader = DataLoader(valid_dataset, shuffle=False, pin_memory=True,
                            batch_size=args.batch_size,
                            num_workers=0)

        val_pbar = tqdm(enumerate(val_loader), total=len(loader))

        with torch.no_grad():
            dataset_distribution = [0] * train_dataset.categories_num       
            for cur_pbar in [pbar, train_pbar, val_pbar]:
                for it, (x, y) in cur_pbar:
                    x_cond = x.to(device)
                    layouts = x_cond.detach().cpu().numpy()                
            
                    for idx, layout in enumerate(layouts):
                        box_and_label = train_dataset.render_normalized_layout(layout)
                        label = box_and_label[1]
                        for l_perbox in label:
                            dataset_distribution[l_perbox] += 1
            
            # print(dataset_distribution)
        # ppt_dataset_distribution = [680774, 96401, 9452, 1507, 22910, 1338]
        # Rico_dataset_distribution = [13700, 29391, 48726, 35740, 32524, 5815, 1888, 3743, 2551, 19210, 4468, 172, 34]
        # PubLayNet_dataset_distribution = [39738, 5164, 2196, 3592, 4748]

        import matplotlib.pyplot as plt

        name_list = list(train_dataset.component_class.keys())
        # name_list = ['Toolbar', 'Image', 'Text', 'Icon', 'Text\n Button', 'Input', 'List Item', 'Advertisement', 'Pager\n Indicator', 'Web View', 'Background\n Image', 'Drawer', 'Modal']
        plt.figure(figsize=(16, 9))
        
        bar = plt.bar(range(len(dataset_distribution)), dataset_distribution, width=0.6, tick_label=name_list, color='lightcoral')
        plt.bar_label(bar, label_type='edge')
        plt.xticks(rotation=50)
        plt.savefig(os.path.join(samples_dir, f"{args.dataset}_label_distribution.png"))

    def generate_case():
        new_samples_dir = os.path.join(samples_dir, args.generate_type)
        os.makedirs(new_samples_dir, exist_ok=True)

        loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                            batch_size=args.batch_size,
                            num_workers=0)
        test_file = test_dataset.iou_data["file_idx"]

        pbar = tqdm(enumerate(loader), total=len(loader))
        with torch.no_grad():
            for itr, (x, y) in pbar:
                x_cond = x.to(device)
        #         # original image
                layouts = x_cond.detach().cpu().numpy()                
                for idx, layout in enumerate(layouts):
                    box_and_label = test_dataset.render_normalized_layout(layout)
                    if len(box_and_label[1]) < args.min_box_num:
                        continue
                    
                    if args.generate_type == "completion_generate":
                        layout = test_dataset.render(layout[:14])
                    else:
                        layout = test_dataset.render(layout)
                    cur_iou_idx = itr*args.batch_size + idx
                    layout.save(os.path.join(new_samples_dir, f'{test_file[cur_iou_idx]}_original.png'))

                # run three times to find better results
                for k in range(3):
                    if args.generate_type == "category_generate":
                        layouts = sample(model, x_cond[:, :1], steps=test_dataset.max_length,
                                        temperature=1.0, sample=True, top_k=5, only_label=True, gt=x_cond).detach().cpu().numpy()
                    elif args.generate_type == "completion_generate":
                        layouts = sample(model, x_cond[:, :14], steps=test_dataset.max_length,
                                        temperature=1.0, sample=True, top_k=5, only_label=False, gt=x_cond).detach().cpu().numpy()
                    elif args.generate_type == "random_generate":
                        layouts = sample(model, x_cond[:, :1], steps=test_dataset.max_length,
                                        temperature=1.0, sample=True, top_k=5, only_label=False, gt=x_cond).detach().cpu().numpy()
                    
                    for idx, layout in enumerate(layouts):
                        cur_idx = itr*args.batch_size + idx                 
                        box_and_label = test_dataset.render_normalized_layout(layout)
                        if len(box_and_label[1]) < args.min_box_num:
                            continue
                        layout = test_dataset.render(layout)
                        layout.save(os.path.join(new_samples_dir, f'{test_file[cur_idx]}_{k}_ours.png'))
                        torch.save(box_and_label[0], os.path.join(new_samples_dir, f'{test_file[cur_idx]}_{k}_ours.pt'))

                if itr >= args.save_num:
                    break

    def cal_iou():  
        file_names = os.listdir(args.select_samples_dir)
        p = Pool(40)

        for file in file_names:
            file_n = file.split(".")[0]
            if file_n.find("iou") != -1:
                continue
            box_and_label = torch.load(os.path.join(new_samples_dir, f'{file_n}.pt'))
            func = partial(compute_iou, box_and_label)
            
            if args.is_train_iou:
                results = p.map(func, train_bbox_idx)
            else:
                results = p.map(func, generated_bbox_idx)
            temp_ids, temp_ious = map(list, zip(*results))

            temp_ids_s =  [y for _,y in sorted(zip(temp_ious,temp_ids), reverse =True)]
            # temp_ious_s = [x for x,_ in sorted(zip(temp_ious,temp_ids), reverse =True)]

            if args.is_train_iou:
                iou_layout = train_dataset.data[train_file2bboxidx[temp_ids_s[0]]]
                iou_layout = train_dataset.render(iou_layout)
                iou_layout.save(os.path.join(args.select_samples_dir, f'{file_n}_train_iou.png'))
            else:    
                source_path = os.path.join(new_samples_dir, f"{temp_ids_s[1]}.png")
                destination_path = os.path.join(args.select_samples_dir, f"{file_n}_gen{temp_ids_s[1]}_iou.png")
                shutil.copy(source_path, destination_path)

        atexit.register(p.close)

    def compute_iou(box_1, idx_2_box):
        # box_1: [N, 4]  box_2: [N, 4]
        if isinstance(box_1, np.ndarray):
            lib = np
        elif isinstance(box_1, torch.Tensor):
            lib = torch
        else:
            raise NotImplementedError(type(box_1))

        if args.is_train_iou:
            box_2 = train_bbox[idx_2_box]
            file_idx = train_dataset.iou_data["file_idx"][idx_2_box]
        else:
            box_2 = generated_bbox[idx_2_box]
            file_idx = generated_idx_to_file[idx_2_box]

        if len(box_2.shape) == 3:
            box_2 = box_2.squeeze(0)
        
        box_1 = add_zeros_element(box_1, 9)
        box_2 = add_zeros_element(box_2, 9)

        l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
        l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)    
        a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

        # intersection
        l_max = lib.maximum(l1, l2)
        r_min = lib.minimum(r1, r2)
        t_max = lib.maximum(t1, t2)
        b_min = lib.minimum(b1, b2)
        cond = (l_max < r_min) & (t_max < b_min)
        ai = lib.where(cond, (r_min - l_max) * (b_min - t_max),
                    lib.zeros_like(a1[0]))

        au = a1 + a2 - ai
        n_class_union = (np.sum(au) > 0).sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            iou_c = np.true_divide(ai,au) 
            iou_c[iou_c == np.inf] = 0
            iou = np.nan_to_num(iou_c)

        iou = np.sum(iou)/n_class_union
        return file_idx, iou
        
    if args.command == "save_platte":
        save_platte_color()
    elif args.command == "dataset_distribution":
        cal_dataset_distribution()
    elif args.command == "generate_ours_case":
        generate_case()
    elif args.command == "cal_iou":
        new_samples_dir = os.path.join(samples_dir, args.generate_type)
        os.makedirs(new_samples_dir, exist_ok=True)
        
        # generated bbox
        if not args.is_train_iou:
            file_list = os.listdir(new_samples_dir)
            generated_bbox = []
            generated_idx_to_file = {}
            generated_file_to_idx = {}
            bbox_idx = 0
            for file in file_list:
                if file.split(".")[-1] == "pt":
                    box_and_label = torch.load(os.path.join(new_samples_dir, file))
                    generated_bbox.append(box_and_label)
                    generated_idx_to_file[bbox_idx] = file.split(".")[0]
                    generated_file_to_idx[file.split(".")[0]] = bbox_idx
                    bbox_idx += 1
            generated_bbox_idx = list(range(len(generated_bbox)))
        else:
            train_dataset = get_dataset(args.dataset, "train")
            train_bbox = train_dataset.iou_data["bbox"]
            train_bbox_idx = list(range(len(train_bbox)))
            train_file2bboxidx = train_dataset.iou_data["file2bboxidx"]
        cal_iou()