from genericpath import exists
import os
import argparse
from tkinter.messagebox import NO
os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from datas import get_dataset
from metric import LayoutFID, compute_maximum_iou
from models.layoutganpp import Generator, Discriminator
from datas.util import LexicographicSort, HorizontalFlip, GaussianNoise, VerticalFlip
from util import init_experiment, save_image, save_checkpoint, convert_layout_to_image
from tqdm import tqdm

def main(args):

    out_dir = init_experiment(args, "FIDNet")
    real_img_dir = "datas/dataset/ppt/raw/image/Real"
    false_img_dir = "datas/dataset/ppt/raw/image/val_false"
    os.makedirs(real_img_dir, exist_ok=True)
    os.makedirs(false_img_dir, exist_ok=True)

    writer = SummaryWriter(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    transforms = [LexicographicSort()]

    # if args.aug_flip:
        # transforms = [T.RandomApply([HorizontalFlip()], 0.5)] + transforms
        # transforms = [T.RandomApply([GaussianNoise()], 0.5)] + transforms
    # transforms_fake = T.RandomChoice([VerticalFlip(), HorizontalFlip(), GaussianNoise()])
    transforms_fake = GaussianNoise()

    train_dataset = get_dataset(args.dataset, 'train',
                                transform=T.Compose(transforms))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)

    val_dataset = get_dataset(args.dataset, 'val')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=False)

    num_label = train_dataset.num_classes

    # setup model
    netD = Discriminator(num_label,
                         d_model=args.D_d_model,
                         nhead=args.D_nhead,
                         num_layers=args.D_num_layers,
                         ).to(device)

    # prepare for evaluation
    # fid_val = LayoutFID()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    Height = 256
    Width = 256


    iteration = 0
    last_eval = -1e+8
    best_FID = 1000
    max_epoch = args.iteration * args.batch_size / len(train_dataset)
    max_epoch = int(torch.ceil(torch.tensor(max_epoch)).item())
    max_epoch = 1
    for epoch in range(max_epoch):
        netD.train()
        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            data = data.to(device)
            label, mask = to_dense_batch(data.y, data.batch)
            bbox_real, _ = to_dense_batch(data.x, data.batch)
            padding_mask = ~mask

            # for j in range(bbox_real.size(0)):
            #     mask_j = mask[j]
            #     b = bbox_real[j][mask_j].cpu().numpy()
            #     l = label[j][mask_j].cpu().numpy()

            #     idx = i * args.batch_size + j
            #     convert_layout_to_image(
            #         b, l, train_dataset.colors, (Height, Width)
            #         ).save(os.path.join(real_img_dir, f'{idx}.png'))

            if args.negative_case == "only_shuffle":
            # only shuffle bbox
                sorted_idx = torch.randperm(data.batch.shape[0]).to(device)
                back_x = data.x[sorted_idx]
                back_y = data.y[sorted_idx]

                sorted_box, _ = to_dense_batch(back_x, data.batch)
                sorted_label, sorted_mask = to_dense_batch(back_y, data.batch)
                sorted_padding_mask = ~sorted_mask

            elif args.negative_case == "gauss_shuffle":
                ele_num = int(0.5 * data.batch[-1].item())
                bs_idx = (data.batch == ele_num).nonzero(as_tuple=True)[0][0].item()
                front_clone = data.x.clone()
                front_clone.x = data.x[:bs_idx]
                front_data = transforms_fake(front_clone)
                front_bbox, _ = to_dense_batch(front_data.x, data.batch[:bs_idx])

                # shuffle bbox
                sorted_idx = torch.randperm(data.batch.shape[0]-bs_idx).to(device)
                back_x = data.x[bs_idx:][sorted_idx]
                back_y = data.y[bs_idx:][sorted_idx]

                back_bbox, _ = to_dense_batch(back_x, data.batch[bs_idx:]-ele_num)
                back_label, back_mask = to_dense_batch(back_y, data.batch[bs_idx:]-ele_num)
                
                f_bs, f_max_num, _ = front_bbox.shape
                b_bs, b_max_num, _ = back_bbox.shape
                if f_max_num < b_max_num:
                    padding = torch.zeros(f_bs, b_max_num-f_max_num, 4).to(device)
                    front_bbox = torch.cat((front_bbox, padding), dim=1)
                elif f_max_num > b_max_num:
                    bbox_padding = torch.zeros(b_bs, f_max_num-b_max_num, 4).to(device)
                    back_bbox = torch.cat((back_bbox, bbox_padding), dim=1)
                    label_padding = torch.zeros(b_bs, f_max_num-b_max_num).to(device)
                    back_label = torch.cat((back_label, label_padding), dim=1).to(dtype=torch.long)
                    back_mask= torch.cat((back_mask, label_padding), dim=1).to(dtype=torch.bool)

                sorted_box = torch.cat((front_bbox, back_bbox), dim=0)
                sorted_label = torch.cat((label[:front_bbox.shape[0]], back_label), dim=0)
                sorted_mask = torch.cat((mask[:front_bbox.shape[0]], back_mask), dim=0)
                sorted_padding_mask = ~sorted_mask

            for j in range(sorted_box.size(0)):
                mask_j = sorted_mask[j]
                b = sorted_box[j][mask_j].cpu().numpy()
                l = sorted_label[j][mask_j].cpu().numpy()

                idx = i * args.batch_size + j
                convert_layout_to_image(
                    b, l, train_dataset.colors, (Height, Width)
                    ).save(os.path.join(false_img_dir, f'{idx}.png'))


            # Update D network
        #     netD.zero_grad()
        #     D_fake = netD(sorted_box, sorted_label, sorted_padding_mask)
        #     loss_D_fake = F.softplus(D_fake).mean()

        #     D_real, logit_cls, bbox_recon = \
        #         netD(bbox_real, label, padding_mask, reconst=True)
        #     loss_D_real = F.softplus(-D_real).mean()
            
        #     loss_D_recl = F.cross_entropy(logit_cls, data.y)
        #     loss_D_recb = F.mse_loss(bbox_recon, data.x)

        #     loss_D = loss_D_real + loss_D_fake
        #     loss_D += loss_D_recl + 10 * loss_D_recb
        #     loss_D.backward()
        #     optimizerD.step()

        #     if iteration % 50 == 0:
        #         D_real = torch.sigmoid(D_real).mean().item()
        #         D_fake = torch.sigmoid(D_fake).mean().item()
        #         loss_D = loss_D.item()
        #         loss_D_fake, loss_D_real = loss_D_fake.item(), loss_D_real.item()
        #         loss_D_recl, loss_D_recb = loss_D_recl.item(), loss_D_recb.item()

        #         print('\t'.join([
        #             f'[{epoch}/{max_epoch}][{i}/{len(train_dataloader)}]',
        #             f'Loss_D: {loss_D:E}', 
        #             f'Real: {D_real:.3f}', f'Fake: {D_fake:.3f}',
        #         ]))

        #         # add data to tensorboard
        #         tag_scalar_dict = {'real': D_real, 'fake': D_fake}
        #         writer.add_scalars('Train/D_value', tag_scalar_dict, iteration)
        #         writer.add_scalar('Train/Loss_D', loss_D, iteration)
        #         writer.add_scalar('Train/Loss_D_fake', loss_D_fake, iteration)
        #         writer.add_scalar('Train/Loss_D_real', loss_D_real, iteration)
        #         writer.add_scalar('Train/Loss_D_recl', loss_D_recl, iteration)
        #         writer.add_scalar('Train/Loss_D_recb', loss_D_recb, iteration)

        #     iteration += 1

        # if epoch != max_epoch - 1:
        #     if iteration - last_eval < 1e+4:
        #         continue

        # # validation
        # last_eval = iteration
        # netD.eval()
        # with torch.no_grad():
        #     for i, data in enumerate(val_dataloader):
        #         data = data.to(device)
        #         label, mask = to_dense_batch(data.y, data.batch)
        #         bbox_real, _ = to_dense_batch(data.x, data.batch)
        #         padding_mask = ~mask

        #         fid_val.collect_features(netD, bbox_real, label, padding_mask)

        #     for i, data in enumerate(train_dataloader):
        #         data = data.to(device)
        #         label, mask = to_dense_batch(data.y, data.batch)
        #         bbox_real, _ = to_dense_batch(data.x, data.batch)
        #         padding_mask = ~mask

        #         fid_val.collect_features(netD, bbox_real, label, padding_mask,
        #                                  real=True)

        # fid_score_val = fid_val.compute_score()

        # writer.add_scalar('Epoch', epoch, iteration)
        # tag_scalar_dict = {'val': fid_score_val}
        # writer.add_scalars('Score/Layout FID', tag_scalar_dict, iteration)

        # print("fid_score_val: ", fid_score_val.item())
        # if fid_score_val.item() < best_FID:
        #     best_FID = fid_score_val.item()
        #     is_best = True
        # else:
        #     is_best = False

        # save_checkpoint({
        #     'args': vars(args),
        #     'epoch': epoch + 1,
        #     'netD': netD.state_dict(),
        #     'optimizerD': optimizerD.state_dict(),
        # }, is_best, out_dir)

def cal_acc(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    transforms = [LexicographicSort()]

    # if args.aug_flip:
        # transforms = [T.RandomApply([HorizontalFlip()], 0.5)] + transforms
        # transforms = [T.RandomApply([GaussianNoise()], 0.5)] + transforms
    transforms_fake = T.RandomChoice([VerticalFlip(), HorizontalFlip(), GaussianNoise()])

    test_dataset = get_dataset(args.dataset, 'test',
                                transform=T.Compose(transforms))
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)

    num_label = test_dataset.num_classes

    # setup model
    netD = Discriminator(num_label,
                         d_model=args.D_d_model,
                         nhead=args.D_nhead,
                         num_layers=args.D_num_layers,
                         ).to(device)

    if args.model_path is not None:
        netD.load_state_dict(torch.load(args.model_path)["netD"])
    else:
        print("model path does not be provided.")

    
    disc_true = []
    disc_false = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            data = data.to(device)
            label, mask = to_dense_batch(data.y, data.batch)
            bbox_real, _ = to_dense_batch(data.x, data.batch)
            padding_mask = ~mask

            D_true = netD(bbox_real, label, padding_mask)
            disc_true.append(D_true)

            if args.negative_case == "only_shuffle":
                # only shuffle bbox
                sorted_idx = torch.randperm(data.batch.shape[0]).to(device)
                back_x = data.x[sorted_idx]
                back_y = data.y[sorted_idx]

                sorted_box, _ = to_dense_batch(back_x, data.batch)
                sorted_label, sorted_mask = to_dense_batch(back_y, data.batch)
                sorted_padding_mask = ~sorted_mask

            elif args.negative_case == "gauss_shuffle":
                ele_num = int(0.5 * data.batch[-1].item())
                bs_idx = (data.batch == ele_num).nonzero(as_tuple=True)[0][0].item()
                front_clone = data.x.clone()
                front_clone.x = data.x[:bs_idx]
                front_data = transforms_fake(front_clone)
                front_bbox, _ = to_dense_batch(front_data.x, data.batch[:bs_idx])

                # shuffle bbox
                sorted_idx = torch.randperm(data.batch.shape[0]-bs_idx).to(device)
                back_x = data.x[bs_idx:][sorted_idx]
                back_y = data.y[bs_idx:][sorted_idx]

                back_bbox, _ = to_dense_batch(back_x, data.batch[bs_idx:]-ele_num)
                back_label, back_mask = to_dense_batch(back_y, data.batch[bs_idx:]-ele_num)
                
                f_bs, f_max_num, _ = front_bbox.shape
                b_bs, b_max_num, _ = back_bbox.shape
                if f_max_num < b_max_num:
                    padding = torch.zeros(f_bs, b_max_num-f_max_num, 4).to(device)
                    front_bbox = torch.cat((front_bbox, padding), dim=1)
                elif f_max_num > b_max_num:
                    bbox_padding = torch.zeros(b_bs, f_max_num-b_max_num, 4).to(device)
                    back_bbox = torch.cat((back_bbox, bbox_padding), dim=1)
                    label_padding = torch.zeros(b_bs, f_max_num-b_max_num).to(device)
                    back_label = torch.cat((back_label, label_padding), dim=1).to(dtype=torch.long)
                    back_mask= torch.cat((back_mask, label_padding), dim=1).to(dtype=torch.bool)

                sorted_box = torch.cat((front_bbox, back_bbox), dim=0)
                sorted_label = torch.cat((label[:front_bbox.shape[0]], back_label), dim=0)
                sorted_mask = torch.cat((mask[:front_bbox.shape[0]], back_mask), dim=0)
                sorted_padding_mask = ~sorted_mask            

            D_fake = netD(sorted_box, sorted_label, sorted_padding_mask)
            disc_false.append(D_fake)

        disc_true = torch.cat(disc_true, dim=0).cpu().detach().numpy().reshape(-1)
        disc_false = torch.cat(disc_false, dim=0).cpu().detach().numpy().reshape(-1)
        
        predict = np.concatenate((disc_true, disc_false), axis=0)
        predict[predict>0] = 1
        predict[predict<0] = 0
        target = np.concatenate((np.ones(disc_true.shape), np.zeros(disc_false.shape)), axis=0)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(target, predict)
        print("acc: ", acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--name', type=str, default='',
                        help='experiment name')
    parser.add_argument('--dataset', type=str, default='rico',
                        choices=['rico', 'publaynet', 'magazine', 'ppt'],
                        help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--iteration', type=int, default=int(2e+5),
                        help='number of iterations to train for')
    parser.add_argument('--seed', type=int, help='manual seed')

    # General
    parser.add_argument('--latent_size', type=int, default=4,
                        help='latent size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--aug_flip', action='store_true',
                        help='use horizontal flip for data augmentation.')

    # Discriminator
    parser.add_argument('--D_d_model', type=int, default=256,
                        help='d_model for discriminator')
    parser.add_argument('--D_nhead', type=int, default=4,
                        help='nhead for discriminator')
    parser.add_argument('--D_num_layers', type=int, default=8,
                        help='num_layers for discriminator')

    # negative case
    parser.add_argument('--negative_case', type=str, default='gauss_shuffle', choices=['only_gauss', 'only_shuffle', 'gauss_shuffle'] )

    # for evaluation like accuracy
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--command', default="cal_acc", choices=['cal_acc', 'train'])

    args = parser.parse_args()

    if args.command == "train":
        main(args)
    elif args.command == "cal_acc":
        cal_acc(args)
