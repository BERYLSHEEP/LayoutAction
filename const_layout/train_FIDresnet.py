
from email.policy import default
import os
import argparse
from random import choices
from metric import calculate_frechet_distance
import numpy as np
import pickle
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.resnet import ImgNet
from data_loader import make_dataset, get_data_loader
from util import init_experiment, save_image, save_checkpoint, convert_layout_to_image


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--name', type=str, default='',
                        help='experiment name')
    parser.add_argument('--dataset', type=str, default='rico',
                        choices=['rico', 'publaynet', 'magazine'],
                        help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--iteration', type=int, default=int(2e+5),
                        help='number of iterations to train for')
    parser.add_argument('--seed', type=int, help='manual seed')

    # General
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')

    # Discriminator
    parser.add_argument('--img_dim', type=int, default=256,
                        help='img dimension for discriminator')
    parser.add_argument('--resnet_num', type=int, default=18, choices=[18, 34])

    args = parser.parse_args()
    print(args)

    out_dir = init_experiment(args, "FIDNet")
    real_img_dir = "datas/dataset/rico/raw/image/Real"
    false_img_dir = "datas/dataset/rico/raw/image/False"

    real_img_dataset = make_dataset(real_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    false_img_dataset = make_dataset(false_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    real_data = get_data_loader(real_img_dataset, args.batch_size, num_workers=4)
    false_data = get_data_loader(false_img_dataset, args.batch_size, num_workers=4)

    writer = SummaryWriter(out_dir)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    netD = ImgNet(resnet_num=args.resnet_num).to(device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    iteration = 0
    max_epoch = args.iteration * args.batch_size / len(real_img_dataset)
    max_epoch = int(torch.ceil(torch.tensor(max_epoch)).item())
    for epoch in range(max_epoch):
        netD.train()
        for i, (real_imgs, false_imgs) in enumerate(zip(real_data, false_data)):
            real_imgs = real_imgs.to(device)
            false_imgs = false_imgs.to(device)

            # Update D network
            netD.zero_grad()
            D_fake, fake_features = netD(false_imgs)
            loss_D_fake = F.softplus(D_fake).mean()

            D_real, real_features = netD(real_imgs)
            loss_D_real = F.softplus(-D_real).mean()

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizerD.step()

            if iteration % 50 == 0:
                D_real = torch.sigmoid(D_real).mean().item()
                D_fake = torch.sigmoid(D_fake).mean().item()
                loss_D = loss_D.item()
                loss_D_fake, loss_D_real = loss_D_fake.item(), loss_D_real.item()

                print('\t'.join([
                    f'[{epoch}/{max_epoch}][{i}/{len(real_img_dataset)}]',
                    f'Loss_D: {loss_D:E}', 
                    f'Real: {D_real:.3f}', f'Fake: {D_fake:.3f}',
                ]))

                # add data to tensorboard
                tag_scalar_dict = {'real': D_real, 'fake': D_fake}
                writer.add_scalars('Train/D_value', tag_scalar_dict, iteration)
                writer.add_scalar('Train/Loss_D', loss_D, iteration)
                writer.add_scalar('Train/Loss_D_fake', loss_D_fake, iteration)
                writer.add_scalar('Train/Loss_D_real', loss_D_real, iteration)

            iteration += 1

        save_checkpoint({
            'args': vars(args),
            'epoch': epoch + 1,
            'netD': netD.state_dict(),
            'optimizerD': optimizerD.state_dict(),
        }, False, out_dir)

class LayoutImgFID():
    def __init__(self, resnet_num, model_path, device='cpu'):
        self.model = ImgNet(resnet_num=resnet_num).to(device)

        # load pre-trained LayoutNet
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict["netD"])
        self.model.requires_grad_(False)
        self.model.eval()

        self.real_features = []
        self.fake_features = []

    def collect_features(self, imgs, real=False):
        if real and type(self.real_features) != list:
            return

        D_fake, feats = self.model(imgs)
        features = self.real_features if real else self.fake_features
        features.append(feats.cpu().numpy())

    def compute_score(self):
        feats_1 = np.concatenate(self.fake_features)
        self.fake_features = []

        if type(self.real_features) == list:
            feats_2 = np.concatenate(self.real_features)
            self.real_features = feats_2
        else:
            feats_2 = self.real_features

        mu_1 = np.mean(feats_1, axis=0)
        sigma_1 = np.cov(feats_1, rowvar=False)
        mu_2 = np.mean(feats_2, axis=0)
        sigma_2 = np.cov(feats_2, rowvar=False)

        return calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)

def print_scores(score_dict):
    for k, v in score_dict.items():
        if k in ['Alignment', 'Overlap']:
            v = [_v * 100 for _v in v]
        if len(v) > 1:
            mean, std = np.mean(v), np.std(v)
            print(f'\t{k}: {mean:.2f} ({std:.2f})')
        else:
            print(f'\t{k}: {v[0]:.2f}')

def eval():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--name', type=str, default='',
                        help='experiment name')
    parser.add_argument('--dataset', type=str, default='rico',
                        choices=['rico', 'publaynet', 'magazine'],
                        help='dataset name')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')    
    parser.add_argument('--pkl_path', type=str, default=None)
    parser.add_argument('--save_img_dir', type=str, default="")
    
    # Discriminator
    parser.add_argument('--img_dim', type=int, default=256,
                        help='img dimension for discriminator')
    parser.add_argument('--resnet_num', type=int, default=18, choices=[18, 34])
    parser.add_argument('--model_path', type=str, default="output/rico/FIDNet/20220118114404965048/checkpoint.pth.tar")

    args = parser.parse_args()
    print(args)

    if args.pkl_path is not None:
        if args.save_img_dir is None:
            print("The image save path is not specific")
        elif not os.path.isdir(args.save_img_dir):
            os.makedirs(args.save_img_dir, exist_ok=True)
            with Path(args.pkl_path).open('rb') as fb:
                generated_layouts = pickle.load(fb)

            Height = 256
            Width = 256        
            import seaborn as sns
            _colors = sns.color_palette('husl', n_colors=13)
            colors = [tuple(map(lambda x: int(x * 255), c))
                for c in _colors]
            for i in range(0, len(generated_layouts), args.batch_size):
                i_end = min(i + args.batch_size, len(generated_layouts))

                # get batch from data list
                #print(generated_layouts[i:i_end])
                for j, (b, l) in enumerate(generated_layouts[i:i_end]):
                    l = l.astype(int)
                    idx = i*args.batch_size + j
                    convert_layout_to_image(
                        b, l, colors, (Height, Width)
                        ).save(os.path.join(args.save_img_dir, f'{idx}.png'))
        

    real_img_dir = "datas/dataset/rico/raw/image/test"
    false_img_dir = args.save_img_dir

    real_img_dataset = make_dataset(real_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    real_data = get_data_loader(real_img_dataset, args.batch_size, num_workers=4)
    false_img_dataset = make_dataset(false_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    false_data = get_data_loader(false_img_dataset, args.batch_size, num_workers=4)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    fid_test = LayoutImgFID(args.resnet_num, args.model_path, device)
    
    for real_imgs in real_data:
        real_imgs = real_imgs.to(device)
        fid_test.collect_features(real_imgs, real=True)

    for false_imgs in false_data:
        false_imgs = false_imgs.to(device)
        fid_test.collect_features(false_imgs, real=False)

    fid_score = fid_test.compute_score()
    print_scores({
            'FID': [fid_score]
        })

if __name__ == "__main__":
    #main()
    eval()
