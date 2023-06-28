"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
from cgi import test
import os
import math
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from utils import sample
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score

from utils import trim_tokens

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_iters = 0
    final_iters = 0  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
   
class Eval:
    def __init__(self, model=None, test_dataset=None, val_dataset=None, test_fake_dataset=None, pkl_path=None, config=None, device=None):
        self.model = model
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.test_fake_dataset = test_fake_dataset
        self.pkl_path = pkl_path
        self.compute_real = True
        self.config = config
        
        if device is None:
            self.device='cpu'
            if torch.cuda.is_available():
                self.device = torch.cuda.current_device()
                if model is not None:
                    self.model = torch.nn.DataParallel(self.model).to(self.device)
        else:
            self.device = device
            if model is not None:
                self.model = self.model.to(self.device)  

    def FID_cal(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            raw_model.load_state_dict(torch.load(self.config.model_path))
        else:
            print("args model_path is None")
            return
        loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        pbar = tqdm(enumerate(loader), total=len(loader))
        with torch.no_grad():
            for it, (x, y) in pbar:
                x_cond = x.to(self.device)
                raw_model.collect_features(idx=x_cond, real=True, is_collect=True)

            with Path(self.pkl_path).open('rb') as fb:
                generated_layouts = pickle.load(fb)

            for i in range(0, len(generated_layouts), config.batch_size):
                i_end = min(i + config.batch_size, len(generated_layouts))

                # get batch from data list
                gen_layouts =[]
                for idx, (b, l) in enumerate(generated_layouts[i:i_end]):
                    bbox = torch.tensor(b, dtype=torch.float)
                    label = torch.tensor(l, dtype=torch.long)
                    layout = self.test_dataset.render_token_layout(label, bbox)
                    gen_layouts.append(layout.unsqueeze(0))
                gen_layouts = torch.cat(gen_layouts, dim=0).to(self.device)
                raw_model.collect_features(idx=gen_layouts, real=False, is_collect=True)
            fid_score = raw_model.compute_score()
            print("gen fid score:", fid_score)

        if self.compute_real:
            loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(loader))
            with torch.no_grad():
                for it, (x, y) in pbar:
                    x_cond = x.to(self.device)
                    raw_model.collect_features(idx=x_cond, real=False, is_collect=True)
                fid_score = raw_model.compute_score()
            print("real fid score:", fid_score)

    def cal_acc(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            raw_model.load_state_dict(torch.load(self.config.model_path))
        else:
            print("args model_path is None")
            return
        real_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)
        fake_loader = DataLoader(self.test_fake_dataset, shuffle=False, pin_memory=True,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers)
        acc = self._cal_acc(real_loader, fake_loader, raw_model)
        print("acc: ", acc)

    def _cal_acc(self, real_loader, fake_loader, model):
        disc_true = []
        disc_false = []

        model.eval()
        with torch.no_grad():
            pbar = tqdm(real_loader, total=len(real_loader))
            for (x, y) in pbar:
                x = x.to(self.device)
                logit_disc_true, logit_pred_true, loss_true, real_feats = model(idx=x)
                disc_true.append(logit_disc_true)

            pbar = tqdm(fake_loader, total=len(real_loader))
            for (x, y) in pbar:
                x = x.to(self.device)
                logit_disc_fake, logit_pred_fake, loss_fake, fake_feats = model(idx=x)
                disc_false.append(logit_disc_fake)

        disc_true = torch.cat(disc_true, dim=0).cpu().detach().numpy().reshape(-1)
        disc_false = torch.cat(disc_false, dim=0).cpu().detach().numpy().reshape(-1)
        
        predict = np.concatenate((disc_true, disc_false), axis=0)
        predict[predict>0] = 1
        predict[predict<0] = 0
        target = np.concatenate((np.ones(disc_true.shape), np.zeros(disc_false.shape)), axis=0)

        acc = accuracy_score(target, predict)
        return acc


    def eval(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            raw_model.load_state_dict(torch.load(self.config.model_path))
        else:
            print("args model_path is None")
            return
        loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        pbar = tqdm(enumerate(loader), total=len(loader))
        
        results = []
        with torch.no_grad():
            for it, (x, y) in pbar:
                x_cond = x.to(self.device)
                        
                # sample with the provided label gt
                # layouts = x_cond.detach().cpu().numpy()                
                # # # # input_layouts = [self.train_dataset.render(layout) for layout in layouts]
                # for i, layout in enumerate(layouts):
                #     layout = self.test_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'input_{it:02d}_{i:02d}.png'))

                # _, logits, _ = model(x_cond)
                # probs = F.softmax(logits, dim=-1)
                # _, y = torch.topk(probs, k=1, dim=-1)
                # layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
                # recon_layouts = [self.train_dataset.render(layout) for layout in layouts]
                
                # for i, layout in enumerate(layouts):
                #     layout = self.test_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'recon_{it:02d}_{i:02d}.png'))

                layouts = sample(model, x_cond[:, :1], steps=self.test_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=1, only_label=True, gt=x_cond).detach().cpu().numpy()
                for i, layout in enumerate(layouts):
                    box_and_label = self.test_dataset.render_normalized_layout(layout)
                    layout = self.test_dataset.render(layout)
                    layout.save(os.path.join(self.config.samples_dir, f'label_sample_{it:02d}_{i:02d}.png'))
                    results.append(box_and_label)
            
        # # # save results
        import pickle
        with open(self.config.evaluate_layout_path, 'wb') as fb:
            print(self.config.evaluate_layout_path)
            pickle.dump(results, fb)

    def split_data(self):
        test_dataset_length = len(self.test_dataset.data)

        all_data = self.test_dataset.data + self.val_dataset.data

        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            raw_model.load_state_dict(torch.load(self.config.model_path))
        else:
            print("args model_path is None")
            return
        
        fid_scores = []
        for i in range(50):
            sorted_idx = np.arange(len(all_data))
            np.random.shuffle(sorted_idx)
            sorted_data = np.array(all_data)[sorted_idx]
            self.test_dataset.data = sorted_data[:test_dataset_length]
            self.val_dataset.data = sorted_data[test_dataset_length:]

            loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(loader))
            raw_model.real_features = []
            raw_model.fake_features = []
            with torch.no_grad():
                for it, (x, y) in pbar:
                    x_cond = x.to(self.device)
                    raw_model.collect_features(idx=x_cond, real=True, is_collect=True)        

            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

            pbar = tqdm(enumerate(val_loader), total=len(val_loader))
            with torch.no_grad():
                for it, (x, y) in pbar:
                    x_cond = x.to(self.device)
                    raw_model.collect_features(idx=x_cond, real=False, is_collect=True)
                fid_score = raw_model.compute_score()
                fid_scores.append(fid_score)

        fid_scores = np.array(fid_scores)
        fid_mean = np.mean(fid_scores)
        fid_var = np.var(fid_scores)
        print("real fid scores:", fid_scores, f" fid mean: {fid_mean} fid var: {fid_var}")       

class FID_Trainer:
    def __init__(self, model, train_dataset, train_fake_dataset, test_dataset, test_fake_dataset, val_dataset, config, args):
        self.model = model
        self.train_dataset = train_dataset
        self.train_fake_dataset = train_fake_dataset
        self.test_dataset = test_dataset
        self.test_fake_dataset = test_fake_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            # self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.device = "cuda:2"
            self.model = self.model.to(self.device)

    def save_checkpoint(self, name='checkpoint.pth'):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, name)
        logger.info("saving %s", ckpt_path)
        torch.save(raw_model.state_dict(), ckpt_path)

    def shuffle_x(self, batch_x):
        batch_len = []
        batch_token = []
        for x in batch_x:
            x = x.numpy()
            tokens = trim_tokens(x, self.train_dataset.bos_token, self.train_dataset.eos_token, self.train_dataset.pad_token)
            tokens = tokens[: len(tokens) // 5 * 5].reshape(-1, 5)
            batch_token.append(tokens)
            batch_len.append(len(tokens))

        batch_token = np.concatenate(batch_token, axis=0)
        permutation = list(np.random.permutation(len(batch_token)))
        shuffle_x = batch_token[permutation, :]
        out_batch_x = []
        out_batch_y = []
        prev = 0
        for i in batch_len:
            new_x = shuffle_x[prev:prev+i]
            layout = torch.tensor(new_x, dtype=torch.long)
            layout = self.train_dataset.transform(layout.reshape(-1))
            out_batch_x.append(layout['x'].unsqueeze(0))
            out_batch_y.append(layout['y'].unsqueeze(0))
            prev += i
        out_batch_x = torch.cat(out_batch_x, dim=0)
        out_batch_y = torch.cat(out_batch_y, dim=0)
        return out_batch_x, out_batch_y

    def _cal_FID(self, raw_model, test_loader, val_loader, epoch = 0):
        if epoch == 0:
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            with torch.no_grad():
                for it, (x, y) in pbar:
                    x_cond = x.to(self.device)
                    raw_model.collect_features(idx=x_cond, real=True, is_collect=True)

        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        with torch.no_grad():
            for it, (x, y) in pbar:
                x_cond = x.to(self.device)
                raw_model.collect_features(idx=x_cond, real=False, is_collect=True)
            fid_score = raw_model.compute_score()
        print("fid scores: ", fid_score)
        return fid_score


    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            print("load pre-trained model")
            raw_model.load_state_dict(torch.load(self.config.model_path))

        optimizer = raw_model.configure_optimizers(config)
        pad_token = self.train_dataset.vocab_size - 1
        evaler = Eval(device=self.device)

        TripletLoss = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')

        true_loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)
        fake_loader = DataLoader(self.train_fake_dataset, shuffle=True, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)
        test_loader = DataLoader(self.test_dataset, shuffle=True, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)
        test_fake_loader = DataLoader(self.test_fake_dataset, shuffle=True, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)
        val_loader = DataLoader(self.val_dataset, shuffle=True, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)

        def run_epoch(true_loader, fake_loader):
            model.train()
 
            losses = []
            disc_true = []
            disc_false = []
            pbar = tqdm(enumerate(zip(true_loader, fake_loader)), total=len(true_loader))
            for it, ((x, y),(fake_x, fake_y)) in pbar:

                if epoch == 0:
                    self.fixed_x = x[:min(4, len(x))]
                    self.fixed_y = y[:min(4, len(y))]

                # shuffle_x
                shuffle_x, shuffle_y = self.shuffle_x(x)
                shuffle_x = shuffle_x.to(self.device)
                shuffle_y = shuffle_y.to(self.device)

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                fake_x = fake_x.to(self.device)
                fake_y = fake_y.to(self.device)

                # update G
                with torch.set_grad_enabled(True):
                    # import ipdb; ipdb.set_trace()
                    logit_disc_true, logit_pred_true, loss_true, real_feats = model(idx=x, targets=y, pad_token=pad_token)
                    logit_disc_fake, logit_pred_fake, loss_fake, gauss_feats = model(idx=fake_x, targets=fake_y, pad_token=pad_token)
                    logit_disc_f_shuffle, logit_pred_f_shuffle, loss_f_shuffle, shuffle_feats = model(idx=shuffle_x, targets=shuffle_y, pad_token=pad_token)
                    
                    disc_false_loss = F.softplus(logit_disc_fake).mean()
                    disc_f_shuffle_loss = F.softplus(logit_disc_f_shuffle).mean()
                    disc_true_loss = F.softplus(-logit_disc_true).mean()

                    loss = loss_true.mean() + disc_false_loss + disc_true_loss + disc_f_shuffle_loss  # collapse all losses if they are scattered on multiple gpus
                    
                    if config.add_triplet_loss:
                        triplet_loss = TripletLoss(real_feats, gauss_feats, shuffle_feats)
                        loss += triplet_loss
                
                    losses.append(loss.item())
                    disc_false.append(disc_false_loss.item() + disc_f_shuffle_loss.item())
                    disc_true.append(disc_true_loss.item())

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                self.iters += 1
                # decay the learning rate based on our progress
                if config.lr_decay:
                    # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    if self.iters < config.warmup_iters:
                        # linear warmup
                        lr_mult = float(self.iters) / float(max(1, config.warmup_iters))
                    else:
                        # cosine learning rate decay
                        progress = float(self.iters - config.warmup_iters) / float(max(1, config.final_iters - config.warmup_iters))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                if config.add_triplet_loss:
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, DTrue {logit_disc_true.mean().item():.5f}, DFalse {logit_disc_fake.mean().item():.5f}, triplet {triplet_loss.mean().item():.5f}, lr {lr:e}")
                else:
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, DTrue {logit_disc_true.mean().item():.5f}, DFalse {logit_disc_fake.mean().item():.5f}, lr {lr:e}")

            test_loss = float(np.mean(losses))
            logger.info("test loss: %f", test_loss)
            return test_loss

        best_acc = 0
        best_fid = 1000
        for epoch in range(config.max_epochs):
            run_epoch(true_loader, fake_loader)
            
            acc = evaler._cal_acc(test_loader, test_fake_loader, raw_model)
            print("accuracy: ", acc)
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or acc > best_acc
            if self.config.ckpt_dir is not None and good_model:
                best_acc = acc
                print("best accuracy: ", acc)
                self.save_checkpoint(f"model_best_acc_{epoch}_{acc:.2f}.pth")
            
            fid = self._cal_FID(raw_model, test_loader, val_loader, epoch)
            if fid < best_fid:
                best_fid = fid
                print("best fid:", fid)
                self.save_checkpoint(f"model_best_fid_{epoch}_{fid:.2f}.pth") 

            # sample from the model
            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                # import ipdb; ipdb.set_trace()
                # inputs
                layouts = self.fixed_x.detach().cpu().numpy()
                
                # # input_layouts = [self.train_dataset.render(layout) for layout in layouts]
                if (epoch+1) == self.config.sample_every:
                    for i, layout in enumerate(layouts):
                        layout = self.train_dataset.render(layout)
                        layout.save(os.path.join(self.config.samples_dir, f'input_{epoch:02d}_{i:02d}.png'))

                # # reconstruction                
                with torch.no_grad():
                    x_cond = self.fixed_x.to(self.device)
                    logit_disc, logit_pred, loss, _ = model(x_cond)
                    probs = F.softmax(logit_pred, dim=-1)
                    _, y = torch.topk(probs, k=1, dim=-1)
                    #layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
                    layouts = y[:, :, 0].detach().cpu().numpy()

                for i, layout in enumerate(layouts):
                    layout = self.train_dataset.render(layout)
                    layout.save(os.path.join(self.config.samples_dir, f'recon_{epoch:02d}_{i:02d}.png'))