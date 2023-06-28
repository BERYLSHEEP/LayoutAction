from email.policy import default
import os
import argparse
from fid_score import calculate_frechet_distance
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from resnet import ImgNet
from data_loader import make_dataset, get_data_loader
from util import init_experiment, save_image, save_checkpoint, convert_layout_to_image


def train(args):
    out_dir = init_experiment(args, "FIDNet")

    real_img_dataset = make_dataset(args.real_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    real_data = get_data_loader(real_img_dataset, args.batch_size, num_workers=4)
    false_img_dataset = make_dataset(args.false_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    false_data = get_data_loader(false_img_dataset, args.batch_size, num_workers=4)

    if args.add_triplet_loss == True:
        shuffle_img_dataset = make_dataset(args.shuffle_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
        shuffle_data = get_data_loader(shuffle_img_dataset, args.batch_size, num_workers=4)
        gauss_img_dataset = make_dataset(args.gauss_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
        gauss_data = get_data_loader(gauss_img_dataset, args.batch_size, num_workers=4)

    real_val_img_dataset = make_dataset(args.real_val_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    real_val_data = get_data_loader(real_val_img_dataset, args.batch_size, num_workers=4)
    false_val_img_dataset = make_dataset(args.false_val_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    false_val_data = get_data_loader(false_val_img_dataset, args.batch_size, num_workers=4)

    writer = SummaryWriter(out_dir)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    netD = ImgNet(resnet_num=args.resnet_num).to(device)
    if args.model_path is not None:
        netD.load_state_dict(torch.load(args.model_path)["netD"])

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    iteration = 0
    max_epoch = args.iteration * args.batch_size / len(real_img_dataset)
    max_epoch = int(torch.ceil(torch.tensor(max_epoch)).item())
    bar = tqdm(enumerate(zip(real_data, false_data, shuffle_data, gauss_data)), total=len(real_data)) if args.add_triplet_loss else tqdm(enumerate(zip(real_data, false_data)), total=len(real_data))
    
    TripletLoss = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')

    best_acc = 0
    for epoch in range(max_epoch):
        netD.train()
        for i, imgs in bar:
            if args.add_triplet_loss:
                real_imgs, false_imgs, shuffle_imgs, gauss_imgs = imgs
                shuffle_imgs = shuffle_imgs.to(device)
                gauss_imgs = gauss_imgs.to(device)
            else:
                real_imgs, false_imgs = imgs

            real_imgs = real_imgs.to(device)
            false_imgs = false_imgs.to(device)

            # Update D network
            netD.zero_grad()
            D_fake, fake_features = netD(false_imgs)
            loss_D_fake = F.softplus(D_fake).mean()

            D_real, real_features = netD(real_imgs)
            loss_D_real = F.softplus(-D_real).mean()

            loss_D = loss_D_real + loss_D_fake
            if args.add_triplet_loss:
                D_fake_shuffle, fake_shuffle_feats = netD(shuffle_imgs)
                D_fake_gauss, fake_gauss_feats = netD(gauss_imgs)
                triplet_loss = TripletLoss(real_features, fake_gauss_feats, fake_shuffle_feats)
                loss_D += triplet_loss

            loss_D.backward()
            optimizerD.step()

            if iteration % 50 == 0:
                D_real = torch.sigmoid(D_real).mean().item()
                D_fake = torch.sigmoid(D_fake).mean().item()
                loss_D = loss_D.item()
                loss_D_fake, loss_D_real = loss_D_fake.item(), loss_D_real.item()

                if args.add_triplet_loss:
                    triplet_loss = triplet_loss.mean().item()
                    writer.add_scalar('Triplet loss', triplet_loss, iteration)
                    t_word = f'Triplet: {triplet_loss:.3f}'
                else:
                    t_word = " "

                print('\t'.join([
                    f'[{epoch}/{max_epoch}][{i}/{len(real_img_dataset)}]',
                    f'Loss_D: {loss_D:E}', 
                    f'Real: {D_real:.3f}', f'Fake: {D_fake:.3f}', t_word
                ]))

                # add data to tensorboard
                tag_scalar_dict = {'real': D_real, 'fake': D_fake}
                writer.add_scalars('Train/D_value', tag_scalar_dict, iteration)
                writer.add_scalar('Train/Loss_D', loss_D, iteration)
                writer.add_scalar('Train/Loss_D_fake', loss_D_fake, iteration)
                writer.add_scalar('Train/Loss_D_real', loss_D_real, iteration)

            iteration += 1

        acc = _cal_acc(real_val_data, false_val_data, netD, device)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint({
                'args': vars(args),
                'epoch': epoch + 1,
                'netD': netD.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                }, True, out_dir)
        else:
            save_checkpoint({
                'args': vars(args),
                'epoch': epoch + 1,
                'netD': netD.state_dict(),
                'optimizerD': optimizerD.state_dict(),
            }, False, out_dir)
    print("best_acc:", acc)

class LayoutImgFID():
    def __init__(self, resnet_num, model_path, device='cpu'):
        self.model = ImgNet(resnet_num=resnet_num).to(device)

        # load pre-trained LayoutNet
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict["netD"])
        # self.model.requires_grad_(False)
        # self.model.eval()

        self.real_features = []
        self.fake_features = []

    def collect_features(self, imgs, real=False):
        if real and type(self.real_features) != list:
            return

        D_fake, feats = self.model(imgs)
        features = self.real_features if real else self.fake_features
        features.append(feats.cpu().numpy())
        return D_fake

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

def eval(args):
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
        

    false_img_dir = args.save_img_dir

    real_img_dataset = make_dataset(args.real_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    real_data = get_data_loader(real_img_dataset, args.batch_size, num_workers=4)
    false_img_dataset = make_dataset(false_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    false_data = get_data_loader(false_img_dataset, args.batch_size, num_workers=4)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    fid_test = LayoutImgFID(args.resnet_num, args.model_path, device)
  
    fid_test.model.train()

    # max_num = min(len(real_data), len(false_data))
    max_num = 201
    cur_num = 0
    print(len(real_data))
    print(len(false_data))
    with torch.no_grad():
        fid_test.model.zero_grad()
        
        for real_imgs in tqdm(real_data, total=len(real_data)):
            if cur_num >= max_num:
                break
            else:
                cur_num += 1
            real_imgs = real_imgs.to(device)
            fid_test.collect_features(real_imgs, real=True)
        
        cur_num = 0
        for false_imgs in tqdm(false_data, total=len(false_data)):
            if cur_num >= max_num:
                break
            else:
                cur_num += 1
            false_imgs = false_imgs.to(device)
            fid_test.collect_features(false_imgs, real=False)

        fid_score = fid_test.compute_score()
        print_scores({
                'FID': [fid_score]
            })

def cal_acc(args):
    real_img_dataset = make_dataset(args.real_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    false_img_dataset = make_dataset(args.false_img_dir, is_folder=False, resolution=args.img_dim, is_img=True)
    real_data = get_data_loader(real_img_dataset, args.batch_size, num_workers=4)
    false_data = get_data_loader(false_img_dataset, args.batch_size, num_workers=4)


    #torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")#torch.cuda.current_device()

    netD = ImgNet(resnet_num=args.resnet_num).to(device)
    if args.model_path is not None:
        netD.load_state_dict(torch.load(args.model_path)["netD"])
        print(f"load model from {args.model_path}")

    netD.train()
    # ??? 为什么设置netD.eval()时结果会出错？
    acc = _cal_acc(real_data, false_data, netD, device)

def _cal_acc(real_data, false_data, netD, device):
    disc_true = []
    disc_false = []
    
    fid_test = LayoutImgFID(args.resnet_num, args.model_path, device)

    with torch.no_grad():
        netD.zero_grad()
        pbar = tqdm(real_data, total=len(real_data))
        for real_imgs in pbar:
            real_imgs = real_imgs.to(device)
            D_real = fid_test.collect_features(real_imgs, real=True)
            disc_true.append(torch.sigmoid(D_real))
        
        pbar = tqdm(false_data, total=len(false_data))
        for false_imgs in pbar:
            false_imgs = false_imgs.to(device)
            # D_fake, fake_features = netD(false_imgs)
            D_fake = fid_test.collect_features(false_imgs, real=False)
            disc_false.append(torch.sigmoid(D_fake))

    disc_true = torch.cat(disc_true, dim=0).cpu().detach().numpy().reshape(-1)
    disc_false = torch.cat(disc_false, dim=0).cpu().detach().numpy().reshape(-1)
    
    predict = np.concatenate((disc_true, disc_false), axis=0)
    predict[predict>=0.5] = 1
    predict[predict<0.5] = 0
    target = np.concatenate((np.ones(disc_true.shape), np.zeros(disc_false.shape)), axis=0)

    acc = accuracy_score(target, predict)
    print("acc: ", acc)

    
    fid_score = fid_test.compute_score()
    print_scores({
            'FID': [fid_score]
        })

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser("train", help="train resnet for discrinating real/fake images")
    train_parser.add_argument('--name', type=str, default='',
                        help='experiment name')
    train_parser.add_argument('--dataset', type=str, default='rico',
                        choices=['rico', 'publaynet', 'magazine', 'ppt'],
                        help='dataset name')
    train_parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    train_parser.add_argument('--iteration', type=int, default=int(4e+5),
                        help='number of iterations to train for')
    train_parser.add_argument('--seed', type=int, help='manual seed')
    
    train_parser.add_argument("--real_img_dir", type=str, default="")
    train_parser.add_argument("--false_img_dir", type=str, default="")
    train_parser.add_argument("--real_val_img_dir", type=str, default="")
    train_parser.add_argument("--false_val_img_dir", type=str, default="")
    
    train_parser.add_argument("--add_triplet_loss", action='store_true', default=False)
    train_parser.add_argument("--shuffle_img_dir", type=str, default="")
    train_parser.add_argument("--gauss_img_dir", type=str, default="")

    # General
    train_parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    # Discriminator
    train_parser.add_argument('--img_dim', type=int, default=256,
                        help='img dimension for discriminator')
    train_parser.add_argument('--resnet_num', type=int, default=18, choices=[18, 34])
    train_parser.add_argument('--model_path', type=str, default=None)
    
    train_parser.set_defaults(func=train)

# -------------------------------------------------------------------------------------------------

    eval_parser = subparsers.add_parser("eval", help="evaluate the FID scores given the pre-trained resnet model")

    eval_parser.add_argument('--name', type=str, default='',
                        help='experiment name')
    eval_parser.add_argument('--dataset', type=str, default='rico',
                        choices=['rico', 'publaynet', 'magazine', 'ppt'],
                        help='dataset name')
    eval_parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')    
    eval_parser.add_argument('--pkl_path', type=str, default=None)
    eval_parser.add_argument('--save_img_dir', type=str, default=None)
    eval_parser.add_argument('--real_img_dir', type=str, default="")
    eval_parser.add_argument('--device', type=int, default=3)
    
    # Discriminator
    eval_parser.add_argument('--img_dim', type=int, default=256,
                        help='img dimension for discriminator')
    eval_parser.add_argument('--resnet_num', type=int, default=18, choices=[18, 34])
    eval_parser.add_argument('--model_path', type=str, default=None)
    eval_parser.set_defaults(func=eval)

# -------------------------------------------------------------------------------------------------
    acc_parser = subparsers.add_parser("cal_accuracy", help="calculate the accuracy of the image discriminator")
    # Discriminator
    acc_parser.add_argument('--img_dim', type=int, default=256,
                        help='img dimension for discriminator')
    acc_parser.add_argument('--resnet_num', type=int, default=18, choices=[18, 34])
    acc_parser.add_argument("--real_img_dir", type=str)
    acc_parser.add_argument("--false_img_dir", type=str)
    acc_parser.add_argument("--batch_size", type=int, default=64)
    acc_parser.add_argument("--model_path", type=str)
    acc_parser.set_defaults(func=cal_acc)

    args = parser.parse_args()

    args.func(args)

