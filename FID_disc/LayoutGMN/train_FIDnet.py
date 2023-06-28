import warnings

from torch_sparse import saint_subgraph
warnings.simplefilter("ignore")

import torch
import os
import torch.nn.functional as F
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

from training_scripts.util import get_args
from graph_encoder import GraphEncoder, MLP
from graph_aggregator import GraphAggregator
from gnn import GraphEmbeddingNet
from training_scripts.dataloader_single import RicoLayout, data_input_to_gmn
from fid_score import calculate_frechet_distance
from training_scripts.geometry_feat_util import cal_geometry_feats, build_geometry_graph

def shuffle_graph_data(graph_data, device, size):
    '''
    {'node_geometry_features': node_geometry_feats.to(self.device), #torch.from_numpy(np.ones((n_total_nodes, 5), dtype=np.float32)),
                'node_room_ids': node_room_ids.to(self.device),
                'edge_features': edge_feats.to(self.device), #torch.from_numpy(np.ones((n_total_edges, 8), dtype=np.float32)), #edge_feats.float(),
                'from_idx': torch.from_numpy(np.concatenate(from_idx, axis=0)).long().to(self.device),
                'to_idx': torch.from_numpy(np.concatenate(to_idx, axis=0)).long().to(self.device),
                'graph_idx': torch.from_numpy(np.concatenate(graph_idx, axis=0)).to(self.device),#.long(),
                'n_graphs': len(self.batch_sg)
                }
    '''        
    node_geo_feats = graph_data["node_geometry_features"]
    sorted_idx = torch.randperm(node_geo_feats.shape[0]).to(device)
    sorted_room_ids = graph_data["node_room_ids"][sorted_idx]
    sorted_node_geo_feats = node_geo_feats[sorted_idx]

    # render orignal box
    re_ori_box = sorted_node_geo_feats[:, 0:5]*size
    re_ori_box = re_ori_box[:, 0:4].detach().cpu().numpy()

    re_edge_feats = []
    cur_idx = 0
    cur_box =[]
    for idx, graph_idx in enumerate(graph_data["graph_idx"]):
        if graph_idx == cur_idx:
            cur_box.append(re_ori_box[idx][np.newaxis, :])
        else:
            cur_box = np.concatenate(cur_box, axis=0)
            feats = cal_geometry_feats(cur_box)
            rela = build_geometry_graph(feats)
            re_edge_feats.append(rela["feats"])

            cur_box = []
            cur_idx += 1
            cur_box.append(re_ori_box[idx][np.newaxis, :])

    # last layout      
    cur_box = np.concatenate(cur_box, axis=0)
    feats = cal_geometry_feats(cur_box)
    rela = build_geometry_graph(feats)
    re_edge_feats.append(rela["feats"])
    re_edge_feats = torch.from_numpy(np.concatenate(re_edge_feats, axis=0)).float().to(device)

    new_graph_data = {}
    new_graph_data["node_geometry_features"] = sorted_node_geo_feats
    new_graph_data["node_room_ids"] = sorted_room_ids
    new_graph_data["edge_features"] = re_edge_feats
    new_graph_data["from_idx"] = graph_data["from_idx"]
    new_graph_data["to_idx"] = graph_data["to_idx"]
    new_graph_data["graph_idx"] = graph_data["graph_idx"]
    new_graph_data["n_graphs"] = graph_data["n_graphs"]

    return new_graph_data 


class FID_trainer:
    def __init__(self, model, train_dataset, train_fake_dataset, test_dataset, test_fake_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.train_fake_dataset = train_fake_dataset
        self.test_dataset = test_dataset
        self.test_fake_dataset = test_fake_dataset
        self.val_dataset = val_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, name):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.model_save_path, name)
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self):
        gmn_model, config = self.model, self.config
        gmn_model = gmn_model.module if hasattr(self.model, "module") else gmn_model
        if self.config.model_path != None:
            print("load pre-trained model")
            gmn_model.load_state_dict(torch.load(self.config.model_path))

        gmn_model_params = list(gmn_model.parameters())
        optimizer = torch.optim.Adam(gmn_model_params, lr=config.lr)

        gmn_model.train()
        iteration = 0
        epoch = 0
        max_iter = len(self.train_dataset) // config.batch_size

        os.makedirs(self.config.model_save_path, exist_ok=True)

        evaler = FID_eval(self.config, gmn_model, test_dataset=self.test_dataset, val_dataset=self.val_dataset, test_fake_dataset=self.test_fake_dataset, device=self.device)
        
        TripletLoss = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
        best_acc = 0
        best_fid = 1000

        while True:    
            t_data = self.train_dataset.get_batch(config.batch_size)
            f_data = self.train_fake_dataset.get_batch(config.batch_size)

            t_graph_data = data_input_to_gmn(self.config, self.device, t_data["graph_data"]).pack_batch()
            f_graph_data = data_input_to_gmn(self.config, self.device, f_data["graph_data"]).pack_batch()
            
            gmn_model.zero_grad()
            t_graph_vectors = gmn_model(**t_graph_data)
            f_graph_vectors = gmn_model(**f_graph_data)
            
            D_true = gmn_model.dis_with_graph_emb(t_graph_vectors)
            D_fake = gmn_model.dis_with_graph_emb(f_graph_vectors)
            
            # shuffle
            s_graph_data = shuffle_graph_data(t_graph_data, self.device, self.train_dataset.size)
            s_graph_vectors = gmn_model(**s_graph_data)
            D_s_fake = gmn_model.dis_with_graph_emb(s_graph_vectors)

            
            loss_D_fake = F.softplus(D_fake).mean() + F.softplus(D_s_fake).mean() * 0.1
            loss_D_real = F.softplus(-D_true).mean()
            
            loss_D = loss_D_real + loss_D_fake
            
            if config.add_triplet_loss:
                triplet_loss = TripletLoss(t_graph_vectors, f_graph_vectors, s_graph_vectors)
                loss_D = loss_D + triplet_loss

            loss_D.backward()
            optimizer.step()
            

            if iteration % config.show_log_every == 0:
                D_real_item = D_true.mean().item()
                D_fake_item = D_fake.mean().item()
                loss_D = loss_D.item()
                
                print('\t'.join([
                    f'[{epoch}/{config.epochs}][{iteration}/{max_iter}]',
                    f'Loss_D: {loss_D:E}', 
                    f'Real: {D_real_item:.3f}', f'Fake: {D_fake_item:.3f}',
                ]))

            iteration += 1
            if t_data["wrapped"] == True:
                iteration = 0
                epoch += 1

                if epoch % config.save_network_every == 0 and epoch !=0 :
                    acc = evaler._cal_acc(gmn_model)
                    print("cur accuracy: ", acc, "best accuracy: ", best_acc)
                    # fid = evaler._cal_FID(gmn_model)
                    if acc > best_acc:
                        self.save_checkpoint("model_best_acc.pth")
                        best_acc = acc

                    # if fid < best_fid:
                    #     self.save_checkpoint("model_best_fid.pth")
                    #     best_fid = fid

class FID_eval:
    def __init__(self, config, model=None, test_dataset=None, val_dataset=None, test_fake_dataset=None, device=None):
        self.model = model
        self.config = config
        self.real_features = []
        self.fake_features = []
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.test_fake_dataset = test_fake_dataset
        self.device = device
        if self.model is not None:
            self.model = self.model.to(self.device)

        # self.device = 'cpu'
        # if torch.cuda.is_available():
        #     self.device = torch.cuda.current_device()
        #     self.model = torch.nn.DataParallel(self.model).to(self.device)
            

    def compute_FID_scores(self):
        feats_1 = np.concatenate(self.fake_features)
        self.fake_features = []

        if type(self.real_features) == list:
            feats_2 = np.concatenate(self.real_features)
            self.real_features = feats_2
        else:
            feats_2 = self.real_features

        feats_2[np.isnan(feats_2)] = 0
        feats_1[np.isnan(feats_1)] = 0

        mu_1 = np.mean(feats_1, axis=0)
        sigma_1 = np.cov(feats_1, rowvar=False)
        mu_2 = np.mean(feats_2, axis=0)
        sigma_2 = np.cov(feats_2, rowvar=False)

        return calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)

    def cal_FID(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if self.config.model_path != None:
            raw_model.load_state_dict(torch.load(self.config.model_path))
        else:
            print("args model_path is None")
            return
        self._cal_FID(raw_model)

    def _cal_FID(self, raw_model):
        
        raw_model.eval()
        with torch.no_grad():
            while True:    
                t_data = self.test_dataset.get_batch(self.config.batch_size)
                t_graph_data = data_input_to_gmn(self.config, self.device, t_data["graph_data"]).pack_batch()
                t_graph_vectors = raw_model(**t_graph_data)
                self.real_features.append(t_graph_vectors.detach().cpu().numpy())

                if t_data["wrapped"] == True:
                    break

            # compute real FID scores
            while True:    
                t_data = self.val_dataset.get_batch(self.config.batch_size)
                t_graph_data = data_input_to_gmn(self.config, self.device, t_data["graph_data"]).pack_batch()
                t_graph_vectors = raw_model(**t_graph_data)

                self.fake_features.append(t_graph_vectors.detach().cpu().numpy())

                if t_data["wrapped"] == True:
                    break
            scores = self.compute_FID_scores()
            print("real fid scores: ", scores)

            # self.real_features = []
            # self.fake_features = []
            # return scores 

            with Path(self.config.pkl_path).open('rb') as fb:
                generated_layouts = pickle.load(fb)

            for i in range(0, len(generated_layouts), config.batch_size):
                i_end = min(i + config.batch_size, len(generated_layouts))

                # get batch from data list
                gen_layouts =[]
                for idx, (b, l) in enumerate(generated_layouts[i:i_end]):

                    bbox = np.array(bbox, dtype=np.float32)
                    label = np.array(label, dtype=np.int32)
                    
                    f_graph_data = self.test_dataset.render_graph_data(label, bbox)
                    gen_layouts.append(f_graph_data)
                graph_datas = self.test_dataset.batch_sg(gen_layouts)
                graph_datas = data_input_to_gmn(self.config, self.device, graph_datas).pack_batch()
                f_graph_vectors = raw_model(**graph_datas)
                self.fake_features.append(f_graph_vectors.detach().cpu().numpy())

            scores = self.compute_FID_scores()
            print(f"gen FID scores:{scores:3f}")

    def cal_acc(self):
        model = self.model
        raw_model = model.module if hasattr(self.model, "module") else model
        if self.config.model_path != None:
            raw_model.load_state_dict(torch.load(self.config.model_path))
        else:
            print("args model_path is None")
            return
        acc = self._cal_acc(raw_model)
        print("acc:", acc)

    def _cal_acc(self, raw_model):
        disc_true = []
        disc_false = []

        raw_model.eval()
        with torch.no_grad():
            while True:    
                t_data = self.test_dataset.get_batch(self.config.batch_size)
                t_graph_data = data_input_to_gmn(self.config, self.device, t_data["graph_data"]).pack_batch()
                t_graph_vectors = raw_model(**t_graph_data)
                D_true = raw_model.dis_with_graph_emb(t_graph_vectors)
                
                disc_true.append(D_true)

                f_data = self.test_fake_dataset.get_batch(self.config.batch_size)
                f_graph_data = data_input_to_gmn(self.config, self.device, f_data["graph_data"]).pack_batch()
                f_graph_vectors = raw_model(**f_graph_data)
                D_false = raw_model.dis_with_graph_emb(f_graph_vectors)
                disc_false.append(D_false)
                
                # shuffle
                s_graph_data = shuffle_graph_data(t_graph_data, self.device, self.test_dataset.size)
                s_graph_vectors = raw_model(**s_graph_data)
                D_s_fake = raw_model.dis_with_graph_emb(s_graph_vectors)
                disc_false.append(D_s_fake)     

                if t_data["wrapped"] == True:
                    break
            
        disc_true = torch.cat(disc_true, dim=0).cpu().detach().numpy().reshape(-1)
        disc_false = torch.cat(disc_false, dim=0).cpu().detach().numpy().reshape(-1)
        
        no_nan_t = np.where(np.isnan(disc_true) == False)[0]
        disc_true = disc_true[no_nan_t]
        no_nan_f = np.where(np.isnan(disc_false) == False)[0]
        disc_false = disc_false[no_nan_f]

        predict = np.concatenate((disc_true, disc_false), axis=0)
        predict[predict>0] = 1
        predict[predict<0] = 0
        target = np.concatenate((np.ones(disc_true.shape), np.zeros(disc_false.shape)), axis=0)

        acc = accuracy_score(target, predict)
        return acc

if __name__ == "__main__":
    config = get_args()

    encoder_model =  GraphEncoder(config, node_hidden_sizes=[config.node_geometry_feat_dim, config.node_state_dim],
                                edge_hidden_sizes=[config.edge_feat_dim, int(config.node_state_dim)])

    aggregator_model = GraphAggregator(node_hidden_sizes=[config.graph_rep_dim],
            graph_transform_sizes=[config.graph_rep_dim],
            gated=True,
            aggregation_type='sum')


    message_net = MLP([2*config.node_state_dim+int(config.node_state_dim), config.edge_feat_dim, int(config.node_state_dim), int(config.node_state_dim)])
    reverse_message_net = MLP([2*config.node_state_dim+int(config.node_state_dim), config.edge_feat_dim, int(config.node_state_dim), int(config.node_state_dim)])
    node_update_mlp = MLP([2*config.node_state_dim, config.node_geometry_feat_dim, int(config.node_state_dim), config.node_state_dim])

    gmn_model = GraphEmbeddingNet(
        encoder = encoder_model,
        aggregator=aggregator_model,
        message_net=message_net,
        reverse_message_net=reverse_message_net,
        node_update_MLP=node_update_mlp,
        node_state_dim=config.node_state_dim,
        edge_hidden_sizes=[config.edge_feat_dim, config.node_state_dim * 2,
                                    config.node_state_dim * 2],
        node_hidden_sizes=[config.node_geometry_feat_dim, config.node_state_dim * 2],
        n_prop_layers=config.n_prop_layers,
        share_prop_params=False,
        node_update_type='residual',
        use_reverse_direction=False,
        reverse_dir_param_different=False,
        layer_norm=False
    )

    if config.train_mode:
        train_dataset = RicoLayout(config.train_dir, config)
        train_fake_dataset = RicoLayout(config.train_dir, config, is_fake=True)
        test_dataset = RicoLayout(config.test_dir, config)
        test_fake_dataset = RicoLayout(config.test_dir, config, is_fake=True)
        val_dataset = RicoLayout(config.val_dir, config)        
        
        trainer = FID_trainer(gmn_model, train_dataset, train_fake_dataset, test_dataset, test_fake_dataset, val_dataset, config)
        trainer.train()
    else:
        test_dataset = RicoLayout(config.test_dir, config)
        val_dataset = RicoLayout(config.val_dir, config)
        
        # evaler = FID_eval(config, gmn_model, test_dataset=test_dataset, val_dataset=val_dataset)
        # evaler.cal_FID()

        test_fake_dataset = RicoLayout(config.test_dir, config, is_fake=True)
        evaler = FID_eval(config, gmn_model, test_dataset=test_dataset, test_fake_dataset=test_fake_dataset)
        evaler.cal_acc()