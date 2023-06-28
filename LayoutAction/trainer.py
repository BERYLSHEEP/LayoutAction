import os
import math
import logging

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F

from torch.utils.data.dataloader import DataLoader
from itertools import cycle

from utils import sample

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
    def __init__(self, model, test_dataset, config):
        self.model = model
        self.test_dataset = test_dataset
        self.device=config.device
        self.config = config
        self.model.to(device=self.device)
        # self.device = torch.cuda.current_device()
        # self.model = torch.nn.DataParallel(self.model).to(self.device)       

    def calculate_coverage(self, layout):
        L_layout = np.asarray(layout.convert("L"))
        color_white = np.where(L_layout == 255)[0].shape[0]
        return color_white

    def eval(self, command):
        model = self.model
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            raw_model.load_state_dict(torch.load(self.config.model_path))
        else:
            print("args model_path is None")
            return
        loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)
        print("dataset length:", len(loader))

        pbar = tqdm(enumerate(loader), total=len(loader))
        results = []
        total_box_num = 0
        generate_chosen_num = 0
        copy_chosen_num = 0
        margin_chosen_num = 0
        color_white_num = 0
                
        with torch.no_grad():         
            for it, (x, y) in pbar:
                x_cond = x.to(self.device)
                
                if command["name"] == "random_generate":    
                    layouts = sample(model, x_cond[:, :1], steps=self.test_dataset.max_length,
                                temperature=1.0, sample=True, top_k=5, only_label=False, gt=x_cond).detach().cpu().numpy()
                elif command["name"] == "category_generate":
                    layouts = sample(model, x_cond[:, :1], steps=self.test_dataset.max_length,
                                temperature=1.0, sample=True, top_k=5, only_label=True, gt=x_cond).detach().cpu().numpy()
                elif command["name"] == "real_image":
                    layouts = x_cond.detach().cpu().numpy()
                elif command["name"] == "reconstruction":
                    logits, _ = model(x_cond)
                    probs = F.softmax(logits, dim=-1)
                    _, y = torch.topk(probs, k=1, dim=-1)
                    layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
                # elif command["name"] == "diversity":
                #     repeat_x_cond = x_cond[0, :1].unsqueeze(0).repeat(x_cond.shape[0], 1)
                #     repeat_gt = x_cond[0].unsqueeze(0).repeat(x_cond.shape[0], 1)
                #     layouts = sample(model, repeat_x_cond, steps=self.train_dataset.max_length,
                #                      temperature=1.0, sample=True, top_k=5, only_label=True, gt=repeat_gt).detach().cpu().numpy()
                else:
                    raise ValueError(f"{command['name']} dose not exist.")                    

                for i, layout in enumerate(layouts):
                    if command["save_image"]:
                        layout_img = self.test_dataset.render(layout)
                        layout_img.save(os.path.join(self.config.samples_dir, f'{command["name"]}_{it:02d}_{i:02d}.png'))

                    if command["calculate_coverage"]:
                        layout_img = self.test_dataset.render(layout)
                        color_white = self.calculate_coverage(layout_img)
                        total_box_num += 1
                        color_white_num += color_white

                    if command["save_pkl"]:
                        box_and_label = self.test_dataset.render_normalized_layout(layout)
                        results.append(box_and_label)

                    if command["calculate_probability"]:
                        box_num, generate_chosen, copy_chosen, margin_chosen = self.test_dataset.calculate_prob(layout)
                        total_box_num += box_num
                        generate_chosen_num += generate_chosen
                        copy_chosen_num += copy_chosen
                        margin_chosen_num += margin_chosen                        

        if command["calculate_coverage"]:
            Height = self.test_dataset.H
            Width = self.test_dataset.W
            coverage_rate = color_white_num / (total_box_num * Height * Width)
            print("coverage rate:", coverage_rate)

        if command["calculate_probability"]:
            margin_rate = margin_chosen_num / (total_box_num * 2)
            generate_rate = generate_chosen_num / (total_box_num * 4)
            copy_rate = copy_chosen_num / (total_box_num * 4)
            unnormal_num = total_box_num * 4 - (margin_chosen_num + generate_chosen_num + copy_chosen_num)
            print("margin rate: ", margin_rate)
            print("copy rate: ", copy_rate)
            print("generate rate: ", generate_rate)
            print("unnormal / total: ", unnormal_num , " / ", total_box_num*4)

        if command["save_pkl"]:
            import pickle
            with open(self.config.evaluate_layout_path, 'wb') as fb:
                pickle.dump(results, fb)
        

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, args, train_rela_dataset=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.args = args
        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None
        self.device=config.device
        self.model.to(device=self.device)

        # next step to optimize
        self.train_rela_dataset = train_rela_dataset

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, 'checkpoint.pth')
        logger.info("saving %s", ckpt_path)
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            print("load pre-trained model")
            raw_model.load_state_dict(torch.load(self.config.model_path, map_location=f"cuda:{self.device}"))

        optimizer = raw_model.configure_optimizers(config)
        consistency_criterion = torch.nn.KLDivLoss()

        def run_epoch(split):
            losses = []
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            if self.args.add_shared_representation:
                rela_loader = DataLoader(self.train_rela_dataset, shuffle=True, pin_memory=True,
                                        batch_size=1, num_workers=1
                ) 
                pbar = tqdm(enumerate(zip(cycle(rela_loader), loader))) if is_train else enumerate(loader)
            else:
                pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            
            # for it, (x, y) in pbar:
            for it, load_data in pbar:
                if self.args.add_shared_representation and is_train:
                    ((x1, y1, rela_y), (x, y)) = load_data
                    x1 = x1[0].to(self.device)
                    y1 = y1[0].to(self.device)
                    rela_y = rela_y[0].to(self.device) 
                else:
                    (x, y) = load_data

                if epoch == 0 and not is_train:
                    self.fixed_x = x[:min(4, len(x))].to(self.device)
                    self.fixed_y = y[:min(4, len(y))].to(self.device)

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # import ipdb; ipdb.set_trace()
                    logits, loss = model(idx=x, targets=y, pad_token=self.train_dataset.pad_token)
                    if self.args.add_shared_representation and is_train:
                        lg, ls, features = model(idx=x1, out_features=True)
                        # features shape: [32, 119, 512]
                        consis_losses = 0
                        for pair_idx in range(0, features.shape[0], 2):
                            if (pair_idx+1) == features.shape[0]:
                                a_obj_idx = rela_y[pair_idx]
                                b_obj_idx = rela_y[pair_idx-1]
                                consis_loss = consistency_criterion(F.log_softmax(features[pair_idx][a_obj_idx:a_obj_idx+1], dim=-1), F.softmax(features[pair_idx-1][b_obj_idx:b_obj_idx+1], dim=-1)) + \
                                     consistency_criterion(F.log_softmax(features[pair_idx-1][b_obj_idx:b_obj_idx+1], dim=-1), F.softmax(features[pair_idx][a_obj_idx:a_obj_idx+1], dim=-1))
                            else:
                                a_obj_idx = rela_y[pair_idx]
                                b_obj_idx = rela_y[pair_idx+1]
                                consis_loss = consistency_criterion(F.log_softmax(features[pair_idx][a_obj_idx:a_obj_idx+1], dim=-1), F.softmax(features[pair_idx+1][b_obj_idx:b_obj_idx+1], dim=-1)) + \
                                    consistency_criterion(F.log_softmax(features[pair_idx+1][b_obj_idx:b_obj_idx+1], dim=-1), F.softmax(features[pair_idx][a_obj_idx:a_obj_idx+1], dim=-1))
                            consis_losses += consis_loss
                        consis_losses = consis_losses / ((features.shape[0]+1) // 2)       
                        loss = loss.mean() + consis_losses*self.args.consistence_param
                    else:               
                        loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
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

                    if self.args.add_shared_representation and is_train:
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, consis loss: {consis_losses.item():.5f}, lr {lr:e}")
                    else:
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}, lr {lr:e}")
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                with torch.no_grad():
                    test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_dir is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

            # sample from the model
            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                
                if (epoch+1) == self.config.sample_every:
                    layouts = self.fixed_x.detach().cpu().numpy()
                    for i, layout in enumerate(layouts):
                        layout = self.train_dataset.render(layout)
                        layout.save(os.path.join(self.config.samples_dir, f'input_{i:02d}.png'))
                
                layouts = sample(model, self.fixed_x[:, :1], steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5, only_label=True, gt=self.fixed_x).detach().cpu().numpy()
                for i, layout in enumerate(layouts):
                    layout = self.train_dataset.render(layout)
                    layout.save(os.path.join(self.config.samples_dir, f'category_{epoch:02d}_{i:02d}.png'))