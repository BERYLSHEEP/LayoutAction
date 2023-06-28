import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import json
import os
import random
from .util import gen_colors
from .geometry_feat_util import cal_geometry_feats, build_geometry_graph

class GaussianNoise_D():
    def __call__(self, data, width, height):
        data = data.clone()

        data[:, [2, 3]] = data[:, [2, 3]] - 1
        data[:, [0, 2]] = data[:, [0, 2]] / (width - 1)
        data[:, [1, 3]] = data[:, [1, 3]] / (height - 1)
        data = np.clip(data, 0, 1)

        # add gaussian noise
        std = 0.01
        data = data + torch.normal(0, std=std, size=data.shape).to(device=data.device)
        data = torch.clamp(data, min=0, max=1).to(dtype=torch.float32)

        data[:, [0, 2]] = data[:, [0, 2]] * (width -1)
        data[:, [1, 3]] = data[:, [1, 3]] * (height -1)
        data[:, [2, 3]] = data[:, [2, 3]] + 1
        return data

class RicoLayout(Dataset):
    def __init__(self, data_dir, config, is_fake=False):
        # component_class = {'Text':0, 'Icon':1, 'Image':2, 'Text Button':3, 'Toolbar':4, 'List Item':5, 'Web View':6, 
        # 'Advertisement':7, 'Input':8, 'Drawer':9, 'Background Image':10, 'Card':11, 'Multi-Tab':12, 'Modal':13, 
        # 'Pager Indicator':14, 'Radio Button':15, 'On/Off Switch':16, 'Slider':17, 'Checkbox':18, 'Map View':19,
        # 'Button Bar':20, 'Video':21, 'Bottom Navigation':22, 'Date Picker':23, 'Number Stepper':24}
        component_class = {'Toolbar':0, 'Image':1, 'Text':2, 'Icon':3, 'Text Button':4, 'Input':5,
        'List Item': 6, 'Advertisement': 7, 'Pager Indicator':8, 'Web View':9, 'Background Image':10,
        'Drawer':11, 'Modal':12}
        self.categories_num = len(component_class.keys())

        if not os.path.exists(data_dir):
            raise ValueError("Data Dir dose not exists:{}".format(data_dir))
        
        self.colors = gen_colors(self.categories_num)
        self.config = config

        #self.fake_transform = T.RandomChoice([VerticalFlip_D(), HorizontalFlip_D(), GaussianNoise_D()])
        self.fake_transform = GaussianNoise_D()
        self.size = pow(2, 8)

        # self.edge_geometry_dir = './data/geometry-directed/'
        # print('\nLoading geometric graphs and features from {}\n'.format(self.edge_geometry_dir))

        #Instantiate the ix
        self.split_ix = {"train": [], "test": [], "val": []}
        self.iterators = {'train': 0,  'test': 0,  'val': 0}

        dir_name = data_dir.split("/")[-1]
        self._prefetch_process = BlobFetcher(dir_name, self, dir_name=='train', num_workers = 4)
        
        # def cleanup():
        #     print('Terminating BlobFetcher')
        #     del self._prefetch_process
        # import atexit
        # atexit.register(cleanup)

        dirs = os.listdir(data_dir)
        self.data = []
        for file in dirs:
            if file.split(".")[-1] == "json":
                file_path = os.path.join(data_dir, file)
                with open(file_path, "r") as f:
                    json_file = json.load(f)

                canvas = json_file["bounds"]
                W, H = float(canvas[2]-canvas[0]), float(canvas[3]-canvas[1])
                if canvas[0]!= 0 or canvas[1]!= 0 or W <= 1000:
                    continue
                elements = self.get_all_element(json_file, [])
                elements = list(filter(lambda e: e["componentLabel"] in component_class, elements))
                
                max_elements_num = 9
                if len(elements) <= 1 or len(elements)>max_elements_num:
                    continue

                id = file.split(".")[0]
                
                self.split_ix[dir_name].append(int(id))
                graph = {"id": id} 

                ann_box = []
                ann_cat = []

                for ele in elements:
                    [x_l, y_t, x_r, y_b] = ele["bounds"]
                    xc = (x_l + x_r) / 2.
                    yc = (y_t + y_b) / 2.
                    w = x_r - x_l
                    h = y_b - y_t

                    if w<=0 or h<=0:
                        continue
                    ann_box.append([xc, yc, w, h])
                    ann_cat.append(component_class[ele["componentLabel"]])

                ann_box = np.array(ann_box)

                if is_fake:
                    ann_box = torch.from_numpy(ann_box)
                    out = self.fake_transform(ann_box, W, H)
                    ann_box = out.numpy()

                ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
                # Sort by ann_box[:, 1], then by ann_box[:, 0]
                ann_box = ann_box[ind]
                
                ann_cat = np.array(ann_cat)
                ann_cat = ann_cat[ind]

                graph["class_id"] = ann_cat
                graph["xywh"] = ann_box
                self.data.append(graph)

    def get_graph_data(self, idx):
        id = self.data[idx]["id"]
        # geometry_path = os.path.join(self.edge_geometry_dir, id + '.npy')
        # rela = np.load(geometry_path, allow_pickle=True)[()] # dict contains keys of edges and feats
        '''
        {'edges': [i, j],
        'feats': feats[i][j] }
        '''

        box = self.data[idx]["xywh"]
        box_feats = self.get_box_feats(box)
        feats = cal_geometry_feats(box)
        rela = build_geometry_graph(feats)

        sg_data = {'obj': self.data[idx]["class_id"], 'box_feats': box_feats, 'rela': rela, 'box':box, 'id':id}
        return sg_data

    def quantize_box(self, boxes, width, height):
        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / (width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()
        # round: 四舍五入

        return boxes.astype(np.int32)

    def get_box_feats(self, box):
        boxes = np.array(box)
        xc, yc, w, h = np.hsplit(boxes, 4)
        W, H = self.size, self.size  # We know the height and weight for all semantic UIs are 2560 and 1400
        '''
        x_min = min([x[0] for x in x1])
        x_max = max([x[0] for x in x2])

        y_min = min([y[0] for y in y1])
        y_max = max([y[0] for y in y2])

        W = x_max - x_min
        H = y_max - y_min
        '''
        box_feats = np.hstack((xc / W, yc / H, w / W, h / H, w * h / (W * H)))
        # box_feats = box_feat / np.linalg.norm(box_feats, 2, 1, keepdims=True)
        return box_feats

    def get_all_element(self, p_dic, elements):
        if "children" in p_dic:
            for i in range(len(p_dic["children"])):
                cur_child = p_dic["children"][i]
                elements.append(cur_child)
                elements = self.get_all_element(cur_child, elements)
        return elements

    def get_batch(self, batch_size):
        result = []
        wrapped = False
        for i in range(batch_size):
            tmp_sg_a, tmp_wrapped = self._prefetch_process.get()
            result.append(tmp_sg_a)

            if tmp_wrapped:
                wrapped = True
                break

        data = {}
        data["graph_data"] = self.batch_sg(result)
        data["wrapped"] = wrapped
        return data

    def batch_sg(self, sg_batch):
        obj_batch = [_['obj'] for _ in sg_batch]
        rela_batch = [_['rela'] for _ in sg_batch]
        #box_batch = [_['box'] for _ in sg_batch]
        
        sg_data = []
        for i in range(len(obj_batch)):
            sg_data.append(dict())

        if self.config.use_box_feats:
            box_feats_batch = [_['box_feats'] for _ in sg_batch]
            #obj_labels = -1*np.ones([max_box_len, 1], dtype = 'int')
            #sg_data['box_feats'] = []
            for i in range(len(box_feats_batch)):
                sg_data[i]['box_feats'] = box_feats_batch[i]
                #obj_labels[:obj_batch[i].shape[0]] = obj_batch[i]
                sg_data[i]['ele_ids'] = obj_batch[i]

            for i in range(len(rela_batch)):
                sg_data[i]['rela_edges'] = rela_batch[i]['edges']
                sg_data[i]['rela_feats'] = rela_batch[i]['feats']
        return sg_data
    
    def render_graph_data(self, label, box):
        # turn 0,1 to original size
        # W = 1400
        # H = 2560
        box[:, [0, 2]] = box[:, [0, 2]]*1400
        box[:, [1, 3]] = box[:, [1, 3]]*2560

        box_feats = self.get_box_feats(box)
        feats = cal_geometry_feats(box)
        rela = build_geometry_graph(feats)

        sg_data = {'obj': label, 'box_feats': box_feats, 'rela': rela, 'box':box}
        return sg_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sg_data = self.get_graph_data(idx)
        return sg_data 

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False, num_workers = 4):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
#        self.config =config
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.num_workers = num_workers

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers= self.num_workers,#1, # 4 is usually enough
                                            worker_init_fn=None,
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.seed(3)
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        # ix0 = int(tmp["id"])
        # ix1 = int(ix)
        # assert int(tmp["id"]) == int(ix), f"ix not equal:{ix0}, {ix1}"

        return tmp , wrapped
     
class SubsetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
        #return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class data_input_to_gmn(object):
    def __init__(self, config, device, batch_sg):
        self.config = config
        self.batch_sg = batch_sg
        self.device = device 
    
    def pack_batch(self):
        """Pack a batch of "batch_graph" into a single `GraphData` instance.

        Returns:
          graph_data: a `GraphData` instance, with node and edge indices properly
            shifted.
        """

        from_idx = []
        to_idx = []
        graph_idx = []

        n_total_nodes = 0
        n_total_edges = 0
        node_geometry_feats = []
        node_room_ids = []
        edge_feats = []

        for i, g in enumerate(self.batch_sg):
            n_nodes = g['box_feats'].shape[0]
            n_edges = g['rela_edges'].shape[0]

            node_geometry_feats.append(torch.from_numpy(g['box_feats']))
            node_room_ids.append(torch.from_numpy(g['ele_ids']))
            edge_feats.append(torch.from_numpy(g['rela_feats']))

            edges = np.array(g['rela_edges'], dtype=np.int32)
            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        node_geometry_feats = torch.cat(node_geometry_feats, dim=0).float()
        node_room_ids = torch.cat(node_room_ids, dim=0).float()
        edge_feats = torch.cat(edge_feats, dim=0).float()

        #'''
        if self.device != 'cpu': #on GPU
            return {'node_geometry_features': node_geometry_feats.to(self.device), #torch.from_numpy(np.ones((n_total_nodes, 5), dtype=np.float32)),
                    'node_room_ids': node_room_ids.to(self.device),
                    'edge_features': edge_feats.to(self.device), #torch.from_numpy(np.ones((n_total_edges, 8), dtype=np.float32)), #edge_feats.float(),
                    'from_idx': torch.from_numpy(np.concatenate(from_idx, axis=0)).long().to(self.device),
                    'to_idx': torch.from_numpy(np.concatenate(to_idx, axis=0)).long().to(self.device),
                    'graph_idx': torch.from_numpy(np.concatenate(graph_idx, axis=0)).to(self.device),#.long(),
                    'n_graphs': len(self.batch_sg)
                    }
        else: #on CPU
            return {'node_geometry_features': node_geometry_feats.to(self.device), #torch.from_numpy(np.ones((n_total_nodes, 5), dtype=np.float32)),
                    'node_room_ids': node_room_ids.to(self.device),
                    'edge_features': edge_feats.to(self.device), #torch.from_numpy(np.ones((n_total_edges, 8), dtype=np.float32)), #edge_feats.float(),
                    # torch.from_numpy(np.ones((n_total_edges, 8), dtype=np.float32)), #edge_feats.float(),
                    'from_idx': torch.from_numpy(np.concatenate(from_idx, axis=0)).long(),#.to(self.device),
                    'to_idx': torch.from_numpy(np.concatenate(to_idx, axis=0)).long(),#.to(self.device),
                    'graph_idx': torch.from_numpy(np.concatenate(graph_idx, axis=0)),#.to(self.device),  # .long(),
                    'n_graphs': len(self.batch_sg)
                    }