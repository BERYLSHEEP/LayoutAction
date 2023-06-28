import pickle
import argparse
import numpy as np

import torch
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch

from datas import get_dataset
from metric import LayoutFID, compute_maximum_iou, \
    compute_overlap, compute_alignment
from util import convert_layout_to_image
import os
from tqdm import tqdm


def average(scores):
    return sum(scores) / len(scores)


def print_scores(score_dict):
    for k, v in score_dict.items():
        if k in ['Alignment', 'Overlap']:
            v = [_v * 100 for _v in v]
        if len(v) > 1:
            mean, std = np.mean(v), np.std(v)
            print(f'\t{k}: {mean:.2f} ({std:.2f})')
        else:
            print(f'\t{k}: {v[0]:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name', default='rico')
                        #choices=['rico', 'publaynet', 'magazine'])
    parser.add_argument('--batch_size', type=int,
                        default=8, help='input batch size')
    parser.add_argument('--out_dir', type=str, default="./datas/dataset/rico/raw/image/shuffle")
    parser.add_argument('--compute_real', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # dataset = get_dataset(args.dataset, 'val')
    # dataloader = DataLoader(dataset,
    #                         batch_size=args.batch_size,
    #                         num_workers=4,
    #                         pin_memory=True,
    #                         shuffle=False)

    # prepare for evaluation
    # fid_test = LayoutFID(args.dataset, device)

    # real layouts
    Height = 120
    Width = 80
    os.makedirs(args.out_dir, exist_ok=True)
    # for i, data in enumerate(dataloader):
    #     data = data.to(device)
    #     label, mask = to_dense_batch(data.y, data.batch)
    #     bbox, _ = to_dense_batch(data.x, data.batch)
    #     padding_mask = ~mask

    #     fid_test.collect_features(bbox, label, padding_mask,
    #                               real=True)

    if args.compute_real:
        dataset = get_dataset(args.dataset, 'test')
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=False)
        results = []

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = data.to(device)

            label, mask = to_dense_batch(data.y, data.batch)
            bbox, _ = to_dense_batch(data.x, data.batch)
            padding_mask = ~mask

            print("bbox:", bbox)
            print("label:", label)

            # add gaussian noise
            var = 0.01
            gauss_x = data.x + torch.normal(0, std=var, size=data.x.shape).to(device=data.x.device)
            gauss_x = torch.clamp(gauss_x, min=0, max=1).to(dtype=torch.float32)
            bbox, _ = to_dense_batch(gauss_x, data.batch)

            # shuffle            
            # sorted_idx = np.arange(data.batch.shape[0])
            # np.random.shuffle(sorted_idx)
            # sorted_x = data.x[sorted_idx]
            # sorted_y = data.y[sorted_idx]
            # bbox, _ = to_dense_batch(sorted_x, data.batch)
            # label, mask = to_dense_batch(sorted_y, data.batch)
            # padding_mask = ~mask

            

            for j in range(bbox.size(0)):
                mask_j = mask[j]
                b = bbox[j][mask_j].cpu().numpy()
                l = label[j][mask_j].cpu().numpy()

                results.append((b, l))

                idx = i*args.batch_size + j
                name = data.attr["name"][j]
                convert_layout_to_image(
                    b, l, dataset.colors, (Height, Width)
                    ).save(os.path.join(args.out_dir, f'{name}_gauss.png'))

        
        # save results
        # import pickle
        # evaluate_layout_path = os.path.join(args.out_dir, "generated_layout.pth") 
        # with open(evaluate_layout_path, 'wb') as fb:
        #     pickle.dump(results, fb)

            # miss some elements
            # element_num = data.batch.shape[0]
            # # miss_num = 4
            # # box_num = data.batch[-1] * miss_num
            # box_num_item = min(200, data.batch[-1].item()*2)
            # choice = np.random.choice(element_num, box_num_item, replace=False)
            
            # select_idx = np.arange(0, element_num)
            # sorted_idx = np.fromiter((x for x in select_idx if x not in choice), dtype=select_idx.dtype)
            # sorted_x = data.x[sorted_idx]
            # sorted_y = data.y[sorted_idx]
            # sorted_batch = data.batch[sorted_idx]
            # bbox, _ = to_dense_batch(sorted_x, sorted_batch)
            # label, mask = to_dense_batch(sorted_y, sorted_batch)
            # padding_mask = ~mask
 
            # random sorted
            # sorted_idx = np.arange(data.batch.shape[0])
            # np.random.shuffle(sorted_idx)
            # sorted_batch = data.batch[sorted_idx]
            # new_sorted_idx = np.argsort(sorted_batch.detach().cpu().numpy())
            # sorted_x = data.x[sorted_idx]
            # new_sorted_x = sorted_x[new_sorted_idx]
            # sorted_y = data.y[sorted_idx]
            # new_sorted_y = sorted_y[new_sorted_idx]
            # bbox, _ = to_dense_batch(new_sorted_x, data.batch)
            # label, mask = to_dense_batch(new_sorted_y, data.batch)


                        
            # shift y/x
            # shift_degree = 20/Width
            # data.x[:, 2] = data.x[:, 2] - shift_degree
            # data.x = torch.clamp(data.x, min=0, max=1).to(dtype=torch.float32)
            # bbox, _ = to_dense_batch(data.x, data.batch)
            # label, mask = to_dense_batch(data.y, data.batch)
            # padding_mask = ~mask
 
            # if i == 0:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            #     for j in range(bbox.size(0)):
            #         mask_j = mask[j]
            #         b = bbox[j][mask_j].cpu().numpy()
            #         l = label[j][mask_j].cpu().numpy()

            #         convert_layout_to_image(
            #             b, l, dataset.colors, (Height, Width)
            #             ).save(os.path.join(out_dir, f'test_{j}.png'))
            # break

        #     fid_test.collect_features(bbox, label, padding_mask)

        # fid_score = fid_test.compute_score()

        # print('Real data:')
        # print_scores({
        #     'FID': [fid_score]
        # })
        # print()

if __name__ == "__main__":
    main()