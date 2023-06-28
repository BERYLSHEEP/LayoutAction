import pickle
import argparse
from pathlib import Path
import os
import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch

from util import set_seed, convert_layout_to_image
from datas import get_dataset
from models.layoutganpp import Generator
import seaborn as sns
import  numpy as np


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str, help='checkpoint path', default="./output/ppt/LayoutGAN++/20220302205919217118/model_best.pth.tar")
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('-o', '--out_path', type=str,
                        default='./output/PPT_layoutGAN++/ppt.pkl',
                        help='output pickle path')
    parser.add_argument('--seed', type=int, help='manual seed')

    parser.add_argument('--save_image', action='store_true', help="save the generated image")
    parser.add_argument('--calculate_coverage', action='store_true', help="calculate the coverage rate")
    parser.add_argument('--save_pkl', action='store_true', 
                        help="save the generated bbox for heuristic metrics (FID, IoU, Align and Overlap)")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    out_path = Path(args.out_path)
    out_dir = out_path.parent
    out_image_dir = os.path.join(out_dir, "samples")
    os.makedirs(out_image_dir, exist_ok=True)
    out_dir.mkdir(exist_ok=True, parents=True)

    # load checkpoint
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = ckpt['args']
    print(train_args)

    # load test dataset
    dataset = get_dataset(train_args['dataset'], 'val')
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)
    num_label = dataset.num_classes
    colors = gen_colors(num_label)

    # setup model and load state
    netG = Generator(train_args['latent_size'], num_label,
                     d_model=train_args['G_d_model'],
                     nhead=train_args['G_nhead'],
                     num_layers=train_args['G_num_layers'],
                     ).eval().to(device)
    netG.load_state_dict(ckpt['netG'])

    results = []
    total_box_num = 0
    color_white_num = 0
    Height = 256
    Width = 256

    
    with torch.no_grad():
        for it, data in enumerate(dataloader):
            data = data.to(device)
            label, mask = to_dense_batch(data.y, data.batch)
            padding_mask = ~mask
            z = torch.randn(label.size(0), label.size(1),
                            train_args['latent_size'], device=device)

            bbox = netG(z, label, padding_mask)
            bbox_real, _ = to_dense_batch(data.x, data.batch)
            # bbox = bbox_real

            for j in range(bbox.size(0)):
                mask_j = mask[j]
                b = bbox[j][mask_j].cpu().numpy()
                l = label[j][mask_j].cpu().numpy()

                if args.save_image:
                    name = data.attr["name"][j]
                    convert_layout_to_image(
                            b, l, colors, (Height, Width)
                        ).save(os.path.join(out_image_dir ,  f'{name}_layoutGAN++.png'))

                if args.save_pkl:
                    results.append((b, l))

                if args.calculate_coverage:
                    img = convert_layout_to_image(
                            b, l, dataset.colors, (Height, Width)
                        )

                    color_white = calculate_coverage(img)
                    total_box_num += 1
                    color_white_num += color_white

    if args.calculate_coverage:
        coverage_rate = color_white_num / (total_box_num * Height * Width)
        print("coverage rate:", coverage_rate)

    # save results
    if args.save_pkl:
        with out_path.open('wb') as fb:
            pickle.dump(results, fb)
        print('Generated layouts are saved at:', args.out_path)

def calculate_coverage(layout):
    L_layout = np.asarray(layout.convert("L"))
    color_white = np.where(L_layout == 255)[0].shape[0]
    return color_white

if __name__ == '__main__':
    main()
