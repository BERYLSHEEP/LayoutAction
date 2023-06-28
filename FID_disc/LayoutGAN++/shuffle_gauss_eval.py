import pickle
import argparse
import numpy as np
import torchvision.transforms as T
import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from datas.util import LexicographicSort
from datas import get_dataset
from util import convert_layout_to_image
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name', default='rico')
                        #choices=['rico', 'publaynet', 'infoppt'])
    parser.add_argument('--batch_size', type=int,
                        default=8, help='input batch size')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default="test", const='split', nargs='?')
    parser.add_argument('--command', choices=['shuffle', 'gauss', 'shuffle_gauss', 'real', 'miss_elements', 'shift'], default="save_shuffle", const="save", nargs='?')
    parser.add_argument('--shift_obj', choices=['x', 'y', 'w', 'h'], default="x", const="x", nargs='?')
    parser.add_argument('--shift_range', type=int, default=1)
    parser.add_argument('--save_pkl', action='store_true', default=False)
    parser.add_argument('--save_img', action='store_true', default=False)
    parser.add_argument('--out_dir', type=str, default="./datas/dataset/rico/raw/image/shuffle")
    args = parser.parse_args()

    # real layouts
    Height = 256
    Width = 256
    os.makedirs(args.out_dir, exist_ok=True)

    transforms = [LexicographicSort()]
    dataset = get_dataset(args.dataset, args.split, transform=T.Compose(transforms))
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)
    results = []

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        label, mask = to_dense_batch(data.y, data.batch)
        bbox, _ = to_dense_batch(data.x, data.batch)

        # add gaussian noise
        if args.command == "gauss":
            var = 0.01
            gauss_x = data.x + torch.normal(0, std=var, size=data.x.shape)
            gauss_x = torch.clamp(gauss_x, min=0, max=1).to(dtype=torch.float32)
            bbox, _ = to_dense_batch(gauss_x, data.batch)
        elif args.command == "shuffle":
            sorted_idx = np.arange(data.batch.shape[0])
            np.random.shuffle(sorted_idx)
            sorted_x = data.x[sorted_idx]
            sorted_y = data.y[sorted_idx]
            bbox, _ = to_dense_batch(sorted_x, data.batch)
            label, mask = to_dense_batch(sorted_y, data.batch)
        elif args.command == "shuffle_gauss":
            from datas.util import GaussianNoise
            transforms_fake = GaussianNoise()
            ele_num = int(0.5 * data.batch[-1].item())
            bs_idx = (data.batch == ele_num).nonzero(as_tuple=True)[0][0].item()
            front_clone = data.x.clone()
            front_clone.x = data.x[:bs_idx]
            front_data = transforms_fake(front_clone)
            front_bbox, _ = to_dense_batch(front_data.x, data.batch[:bs_idx])

            # shuffle bbox
            sorted_idx = torch.randperm(data.batch.shape[0]-bs_idx)
            back_x = data.x[bs_idx:][sorted_idx]
            back_y = data.y[bs_idx:][sorted_idx]

            back_bbox, _ = to_dense_batch(back_x, data.batch[bs_idx:]-ele_num)
            back_label, back_mask = to_dense_batch(back_y, data.batch[bs_idx:]-ele_num)
            
            f_bs, f_max_num, _ = front_bbox.shape
            b_bs, b_max_num, _ = back_bbox.shape
            if f_max_num < b_max_num:
                padding = torch.zeros(f_bs, b_max_num-f_max_num, 4)
                front_bbox = torch.cat((front_bbox, padding), dim=1)
            elif f_max_num > b_max_num:
                bbox_padding = torch.zeros(b_bs, f_max_num-b_max_num, 4)
                back_bbox = torch.cat((back_bbox, bbox_padding), dim=1)
                label_padding = torch.zeros(b_bs, f_max_num-b_max_num)
                back_label = torch.cat((back_label, label_padding), dim=1).to(dtype=torch.long)
                back_mask= torch.cat((back_mask, label_padding), dim=1).to(dtype=torch.bool)

            bbox = torch.cat((front_bbox, back_bbox), dim=0)
            label = torch.cat((label[:front_bbox.shape[0]], back_label), dim=0)
            mask = torch.cat((mask[:front_bbox.shape[0]], back_mask), dim=0)
            
        elif args.command == "elements":
            element_num = data.batch.shape[0]
            box_num_item = min(200, data.batch[-1].item()*2)
            choice = np.random.choice(element_num, box_num_item, replace=False)
            
            select_idx = np.arange(0, element_num)
            sorted_idx = np.fromiter((x for x in select_idx if x not in choice), dtype=select_idx.dtype)
            sorted_x = data.x[sorted_idx]
            sorted_y = data.y[sorted_idx]
            sorted_batch = data.batch[sorted_idx]
            bbox, _ = to_dense_batch(sorted_x, sorted_batch)
            label, mask = to_dense_batch(sorted_y, sorted_batch)
        elif args.command == "shift":
            if args.shift_obj == "x":
                shift_degree = args.shift_range/Width
                dim = 0
            elif args.shift_obj == "y":
                shift_degree = args.shift_range/Height
                dim = 1
            elif args.shift_obj == "w":
                shift_degree = args.shift_range/Width
                dim = 2
            elif args.shift_obj == "h":
                shift_degree = args.shift_range/Height
                dim = 3
            data.x[:, dim] = data.x[:, dim] - shift_degree
            data.x = torch.clamp(data.x, min=0, max=1).to(dtype=torch.float32)
            bbox, _ = to_dense_batch(data.x, data.batch)
            label, mask = to_dense_batch(data.y, data.batch)

        for j in range(bbox.size(0)):
            mask_j = mask[j]
            b = bbox[j][mask_j].cpu().numpy()
            l = label[j][mask_j].cpu().numpy()

            if args.save_pkl:
                results.append((b, l))

            if args.save_img:
                name = data.attr["name"][j]
                convert_layout_to_image(
                    b, l, dataset.colors, (Height, Width)
                    ).save(os.path.join(args.out_dir, f'{name}_{args.command}.png'))

    
    # save results
    if args.save_pkl:
        import pickle
        evaluate_layout_path = os.path.join(args.out_dir, "generated_layout.pth") 
        with open(evaluate_layout_path, 'wb') as fb:
            pickle.dump(results, fb)


