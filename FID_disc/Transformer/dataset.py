import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageOps
import json
import os

import torchvision.transforms as T

from utils import trim_tokens, gen_colors

def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

class LexicographicSort():
    def __call__(self, data):
        assert not data.attr['has_canvas_element']
        # 安装t 进行lexicographicSort排序
        l, t, _, _ = convert_xywh_to_ltrb(data.x.t())
        _zip = zip(*sorted(enumerate(zip(t, l)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data.x_orig, data.y_orig = data.x, data.y
        data.x, data.y = data.x[idx], data.y[idx]
        return data


class HorizontalFlip():
    def __call__(self, data):
        data.x = data.x.clone()
        data.x[:, 0] = 1 - data.x[:, 0]
        return data

class VerticalFlip():
    def __call__(self, data):
        data.x = data.x.clone()
        data.x[:, 1] = 1 - data.x[:, 1]
        return data

class GaussianNoise():
    def __call__(self, data, width, height):
        data.x = data.x.clone()
        # add gaussian noise
        std = 0.01
        data.x = data.x + torch.normal(0, std=std, size=data.x.shape).to(device=data.x.device)
        data.x = torch.clamp(data.x, min=0, max=1).to(dtype=torch.float32)
        return data

class HorizontalFlip_D():
    def __call__(self, data):
        data = data.clone()
        data[:, 0] = 1 - data[:, 0]
        return data

class VerticalFlip_D():
    def __call__(self, data):
        data = data.clone()
        data[:, 1] = 1 - data[:, 1]
        return data

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

class Padding(object):
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.pad_token = vocab_size - 1

    def __call__(self, layout):
        # grab a chunk of (max_length + 1) from the layout

        chunk = torch.zeros(self.max_length+1, dtype=torch.long) + self.pad_token
        # Assume len(item) will always be <= self.max_length:
        chunk[0] = self.bos_token
        chunk[1:len(layout)+1] = layout
        chunk[len(layout)+1] = self.eos_token

        x = chunk[:-1]
        y = chunk[1:]
        return {'x': x, 'y': y}

        # fully reconstruction
        # return {'x': chunk[:-1], 'y': chunk[:-1]}

class MNISTLayout(MNIST):

    def __init__(self, root, train=True, download=True, threshold=32, max_length=None):
        super().__init__(root, train=train, download=download)
        self.vocab_size = 784 + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        self.threshold = threshold
        self.data = [self.img_to_set(img) for img in self.data]
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def __len__(self):
        return len(self.data)

    def img_to_set(self, img):
        fg_mask = img >= self.threshold
        fg_idx = fg_mask.nonzero(as_tuple=False)
        fg_idx = fg_idx[:, 0] * 28 + fg_idx[:, 1]
        return fg_idx

    def render(self, layout):
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        x_coords = layout % 28
        y_coords = layout // 28
        # valid_idx = torch.where((y_coords < 28) & (y_coords >= 0))[0]
        img = np.zeros((28, 28, 3)).astype(np.uint8)
        img[y_coords, x_coords] = 255
        return Image.fromarray(img, 'RGB')

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = self.transform(self.data[idx])
        return layout['x'], layout['y']

class JSONLayout(Dataset):
    def __init__(self, json_path, max_length=None, precision=8):
        with open(json_path, "r") as f:
            data = json.loads(f.read())

        images, annotations, categories = data['images'], data['annotations'], data['categories']
        self.size = pow(2, precision)

        self.categories = {c["id"]: c for c in categories}
        self.colors = gen_colors(len(self.categories))

        self.json_category_id_to_contiguous_id = {
            v: i + self.size for i, v in enumerate([c["id"] for c in self.categories.values()])
        }
        # {1: 256, 2: 257, 3: 258, 4: 259, 5: 260}

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.vocab_size = self.size + len(self.categories) + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        image_to_annotations = {}
        for annotation in annotations:
            image_id = annotation["image_id"]

            if not (image_id in image_to_annotations):
                image_to_annotations[image_id] = []

            image_to_annotations[image_id].append(annotation)

        self.data = []
        for image in images:
            image_id = image["id"]
            height, width = float(image["height"]), float(image["width"])

            if image_id not in image_to_annotations:
                continue

            ann_box = []
            ann_cat = []

            for ann in image_to_annotations[image_id]:
                x, y, w, h = ann["bbox"]
                ann_box.append([x, y, w, h])
                ann_cat.append(self.json_category_id_to_contiguous_id[ann["category_id"]])

            # Sort boxes
            ann_box = np.array(ann_box)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            # Sort by ann_box[:, 1], then by ann_box[:, 0]
            ann_box = ann_box[ind]
       
            if len(ann_box) > 128:
                print("elements num: ", len(ann_box))
                continue

            ann_cat = np.array(ann_cat)
            ann_cat = ann_cat[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)
            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

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

    def __len__(self):
        return len(self.data)

    def render(self, layout):
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 255
        box[:, [2, 3]] = box[:, [2, 3]] / self.size * 256
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        return layout['x'], layout['y']

class RicoLayout(Dataset):
    def __init__(self, data_dir, max_length=None, precision=8, is_fake=False):
        # component_class = {'Text':0, 'Icon':1, 'Image':2, 'Text Button':3, 'Toolbar':4, 'List Item':5, 'Web View':6, 
        # 'Advertisement':7, 'Input':8, 'Drawer':9, 'Background Image':10, 'Card':11, 'Multi-Tab':12, 'Modal':13, 
        # 'Pager Indicator':14, 'Radio Button':15, 'On/Off Switch':16, 'Slider':17, 'Checkbox':18, 'Map View':19,
        # 'Button Bar':20, 'Video':21, 'Bottom Navigation':22, 'Date Picker':23, 'Number Stepper':24}
        component_class = {'Toolbar':0, 'Image':1, 'Text':2, 'Icon':3, 'Text Button':4, 'Input':5,
        'List Item': 6, 'Advertisement': 7, 'Pager Indicator':8, 'Web View':9, 'Background Image':10,
        'Drawer':11, 'Modal':12}
        self.W = 80
        self.H = 120
        self.categories_num = len(component_class.keys())

        if not os.path.exists(data_dir):
            raise ValueError("Data Dir dose not exists:{}".format(data_dir))
        
        self.size = pow(2, precision)
        self.colors = gen_colors(self.categories_num)

        self.json_category_id_to_contiguous_id = {
            i: i + self.size for i in range(self.categories_num)
        }
        
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.vocab_size = self.size + self.categories_num + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        #self.fake_transform = T.RandomChoice([VerticalFlip_D(), HorizontalFlip_D(), GaussianNoise_D()])
        self.fake_transform = GaussianNoise_D()

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
                if len(elements) == 0 or len(elements)>max_elements_num:
                    continue

                ann_box = []
                ann_cat = []

                for ele in elements:
                    [x_l, y_t, x_r, y_b] = ele["bounds"]
                    xc = (x_l + x_r) / 2.
                    yc = (y_t + y_b) / 2.
                    w = x_r - x_l
                    h = y_b - y_t

                    if w<0 or h<0:
                        continue
                    ann_box.append([xc, yc, w, h])
                    ann_cat.append(self.json_category_id_to_contiguous_id[component_class[ele["componentLabel"]]])

                ann_box = np.array(ann_box)

                if is_fake:
                    ann_box = torch.from_numpy(ann_box)
                    out = self.fake_transform(ann_box, W, H)  
                    ann_box = out.numpy()

                # Sort boxes
                # Discretize boxes
                ann_box = self.quantize_box(ann_box, W, H)

                ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
                # Sort by ann_box[:, 1], then by ann_box[:, 0]
                ann_box = ann_box[ind]
                
                ann_cat = np.array(ann_cat)
                ann_cat = ann_cat[ind]

                # Append the categories
                layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

                # Flatten and add to the dataset
                self.data.append(layout.reshape(-1))

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def get_all_element(self, p_dic, elements):
        if "children" in p_dic:
            for i in range(len(p_dic["children"])):
                cur_child = p_dic["children"][i]
                elements.append(cur_child)
                elements = self.get_all_element(cur_child, elements)
        return elements

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

    def __len__(self):
        return len(self.data)

    def render_normalized_layout(self, layout):
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        label = layout[:, 0].astype(np.int32)
        label = label - self.size
        box = box / (self.size - 1)
        
        # box = np.clip(box, 0, 1)
        label[label>self.categories_num] = 0
        label[label<0] = 0
        return (box, label)

    def render_token_layout(self, label, box):
        # box: normalized box  [cur_box_num, 4]
        box = torch.clamp(box, min=0, max=1)
        box = box * (self.size -1)
        label = label + self.size
        layout = torch.cat((label.unsqueeze(1), box), dim=1)
        layout = layout.reshape(-1)
        layout = layout.to(dtype=torch.long)
        layout = self.transform(layout)
        return layout['x']

    def render(self, layout):
        img = Image.new('RGB', (self.W, self.H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        # box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 255
        # box[:, [2, 3]] = box[:, [2, 3]] / self.size * 256
        # box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]
        box = box / (self.size - 1)

        box[:, [0, 2]] = box[:, [0, 2]] * self.W
        box[:, [1, 3]] = box[:, [1, 3]] * self.H
        # xywh to ltrb
        x1s = box[:, 0] - box[:, 2] / 2
        y1s = box[:, 1] - box[:, 3] / 2
        x2s = box[:, 0] + box[:, 2] / 2
        y2s = box[:, 1] + box[:, 3] / 2

        for i in range(len(layout)):
            # x1, y1, x2, y2 = box
            x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
            cat = layout[i][0]
            col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        return layout['x'], layout['y']   

class MedianExpRicoLayout(Dataset):
    def __init__(self, data_dir, max_length=None, precision=8, save_name=None):
        # component_class = {'Text':0, 'Icon':1, 'Image':2, 'Text Button':3, 'Toolbar':4, 'List Item':5, 'Web View':6, 
        # 'Advertisement':7, 'Input':8, 'Drawer':9, 'Background Image':10, 'Card':11, 'Multi-Tab':12, 'Modal':13, 
        # 'Pager Indicator':14, 'Radio Button':15, 'On/Off Switch':16, 'Slider':17, 'Checkbox':18, 'Map View':19,
        # 'Button Bar':20, 'Video':21, 'Bottom Navigation':22, 'Date Picker':23, 'Number Stepper':24}
        component_class = {'Toolbar':0, 'Image':1, 'Text':2, 'Icon':3, 'Text Button':4, 'Input':5,
        'List Item': 6, 'Advertisement': 7, 'Pager Indicator':8, 'Web View':9, 'Background Image':10,
        'Drawer':11, 'Modal':12}
        
        self.W = 80
        self.H = 120
        self.size = pow(2, precision)
        self.no_value_token = self.size

        if save_name == None:
            self.save_name = "medianExpRico_{}_new.pt".format(len(component_class.keys()))
        else:
            self.save_name = save_name
        load_path = os.path.join(data_dir, self.save_name)
        if os.path.exists(load_path):
            print("load dataset.")
            self.load_pt(load_path)
        else:
            dirs = os.listdir(data_dir)
            self.max_elements_num = 9
            self.categories_num = len(component_class.keys())
            self.json_category_id_to_contiguous_id = {
                i: i + self.size + 1 for i in range(self.categories_num)
            }
            self.copy_mode = self.size + 1 + self.categories_num
            self.margin_mode = self.copy_mode + 1
            self.generate_mode = self.margin_mode + 1
            self.option_id = np.array([self.copy_mode, self.margin_mode, self.generate_mode])

            self.obj_id_to_contiguous_id = {
                i: i + self.generate_mode + 1 for i in range(self.max_elements_num)
            }
            self.no_obj_token = self.generate_mode + 1

            self.data = []
            for file in dirs:
                if file.split(".")[-1] == "json":
                    file_path = os.path.join(data_dir, file)
                    with open(file_path, encoding='utf-8') as f:
                        json_file = json.load(f)

                    canvas = json_file["bounds"]
                    W, H = float(canvas[2]-canvas[0]), float(canvas[3]-canvas[1])
                    if canvas[0]!= 0 or canvas[1]!= 0 or W <= 1000:
                        continue
                    elements = self.get_all_element(json_file, [])
                    elements = list(filter(lambda e: e["componentLabel"] in component_class, elements))
                    
                    if len(elements) == 0 or len(elements)> self.max_elements_num:
                        continue
                    
                    ann_box = []
                    ann_cat = []

                    for ele in elements:
                        [x_l, y_t, x_r, y_b] = ele["bounds"]
                        xc = (x_l + x_r) / 2.
                        yc = (y_t + y_b) / 2.
                        w = x_r - x_l
                        h = y_b - y_t

                        if w<0 or h<0:
                            continue
                        ann_box.append([xc, yc, w, h])
                        ann_cat.append(self.json_category_id_to_contiguous_id[component_class[ele["componentLabel"]]])

                    # Sort boxes

                    ann_box = np.array(ann_box)
                    # Discretize boxes
                    ann_box = self.quantize_box(ann_box, W, H)

                    ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
                    # Sort by ann_box[:, 1], then by ann_box[:, 0]
                    ann_box = ann_box[ind]
                    
                    ann_cat = np.array(ann_cat)
                    ann_cat = ann_cat[ind]

                    # xywh to Option Obj Value
                    # the first element must from generate option
                    ann_option = [[self.generate_mode]*4]
                    ann_obj = [[self.no_obj_token]*4]
                    ann_value = [ann_box[0]]
                    
                    for ele_idx in range(1, len(ann_box)):
                        choice_gt, copy_label, margin_label, margin_value = self.get_choice_gt(ann_box[:ele_idx+1])
                        # option :  choice_gt [bs, 4, 3]  -->  ann_option [bs, 4]
                        option_idx = np.argmax(choice_gt, axis=-1)
                        ann_option.append(self.option_id[option_idx])

                        # obj: copy_label [pre_num, 4]  margin_label [pre_num, 4]   ---> ann_obj [4]
                        copy_obj_flip_idx = np.argmax(np.flip(copy_label, 0), axis=0)
                        # if the idx is 1, indicates the last (1 + idx) = 2 obj is selected.
                        margin_label_4dim = np.concatenate((margin_label, np.zeros(margin_label.shape)), axis=1)
                        margin_obj_flip_idx = np.argmax(np.flip(margin_label_4dim, 0), axis=0) 

                        copy_obj_hit_idx = (copy_obj_flip_idx + 1) * choice_gt[:, 0]
                        margin_obj_hit_idx = (margin_obj_flip_idx + 1) * choice_gt[:, 1]
                        obj_idx = copy_obj_hit_idx + margin_obj_hit_idx + self.no_obj_token
                        ann_obj.append(obj_idx)

                        copy_hit_value = choice_gt[:, 0] * self.no_value_token
                        margin_value_x4 = np.concatenate((margin_value[-margin_obj_hit_idx[:2], range(2)][np.newaxis, :], np.zeros((1, 2))), axis=1) 
                        margin_hit_value = choice_gt[:, 1] * margin_value_x4
                        generate_hit_value = choice_gt[:, 2] * ann_box[ele_idx]
                        value_idx = copy_hit_value + margin_hit_value + generate_hit_value
                        ann_value.append(value_idx[0].round().astype(np.int32))

                    ann_option = np.array(ann_option)
                    ann_obj = np.array(ann_obj)
                    ann_value = np.array(ann_value)

                    OOV_idx = np.concatenate([ann_option[:, :, np.newaxis], ann_obj[:, :, np.newaxis], ann_value[:, :, np.newaxis]], axis=-1)
                    layout = np.concatenate([ann_cat[:, np.newaxis], OOV_idx.reshape(-1, 12)], axis=-1)
                    
                    # Flatten and add to the dataset
                    self.data.append(layout.reshape(-1))
                    
        self.colors = gen_colors(self.categories_num)

        self.json_category_id_to_contiguous_id = {
            i: i + self.size for i in range(self.categories_num)
        }
        
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.max_length = max_length
        if self.max_length is None:
            max_length_cxywh = max([len(x) for x in self.data])
            self.max_elements_num =  int(max_length_cxywh/ 13)
            # category + (option + obj + value) * (xywh) = 13
            self.max_length = max_length_cxywh + 2  # bos, eos tokens
        
        self.max_elements_num = 9
        self.vocab_size = self.size + 1 + self.categories_num + 3 + 1 + self.max_elements_num + 3  
        # size, no value, category num, copy, margin, generate, no obj, obj idx, bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1
        self.transform = Padding(self.max_length, self.vocab_size)

        if not os.path.exists(load_path):
            self.save_pt(load_path)
            

    def get_choice_gt(self, bbox_gt):
        # generate, copy, margin
        copy_label = self.get_copy_gt(bbox_gt)
        # [pre_bbox_num, 4]
        copy_choice = np.sum(copy_label, axis=0, dtype=np.bool_)
        copy_is_zero = copy_choice == np.zeros(copy_choice.shape)
        margin_label, margin_value = self.get_margin_gt(bbox_gt)
        # [pre_bbox_num, 2]
        margin_labelx4 = np.concatenate((margin_label, np.zeros(margin_label.shape)), axis=1)
        margin_choice = np.sum(margin_labelx4, axis=0, dtype=np.bool_) * copy_is_zero
        margin_is_zero = margin_choice == np.zeros(margin_choice.shape)

        generate_choice = copy_is_zero*margin_is_zero

        choice_gt = np.concatenate((copy_choice[:, np.newaxis], margin_choice[:, np.newaxis], generate_choice[:, np.newaxis]), axis=1)
        
        # normalize the label
        # copy_label = torch.nn.functional.normalize(copy_label.to(torch.float32), dim=1, p=1)
        # margin_label = torch.nn.functional.normalize(margin_label.to(torch.float32), dim=1, p=1)
        return choice_gt, copy_label, margin_label, margin_value
        # return : [4, 3]

    def get_copy_gt(self, bbox_gt):
        # bbox_gt: [cur_bbox_max_num, 4]
        bbox_cur_num, _ = bbox_gt.shape
        bbox_g = bbox_gt[-1, :][np.newaxis, :]
        bbox_g = np.repeat(bbox_g, bbox_cur_num-1, 0)
        copy_label = bbox_g == bbox_gt[:-1, :]
        return copy_label

        # sum = torch.sum(copy_label, dim=1).type(torch.bool)
        # bbox_g_prob = ~sum
        # output = torch.cat((copy_label, bbox_g_prob.unsqueeze(1)), dim=1).type(torch.float32)
        # return output

    def get_margin_gt(self, bbox_gt):
        # bbox_gt : [cur_bbox_num, 4]
        bbox_cur_num = bbox_gt.shape[0]
        bbox_g = bbox_gt[-1, :][np.newaxis, :]
        bbox_g = np.repeat(bbox_g, bbox_cur_num-1, 0)
        margin_l = bbox_g == bbox_gt[:-1, :]
        margin_label = np.concatenate((margin_l[:, 1][:, np.newaxis], margin_l[:, 0][:, np.newaxis]), axis=1)
        # margin_value label
        # [cur_bbox_num-1, 2]
        margin_value = bbox_g[:, :2]  - bbox_gt[:-1, :2] - 0.5*bbox_gt[:-1, 2:] - 0.5*bbox_g[:, 2:]
        # if the margin_value < 0: ---> then the corresponding margin label is False.
        is_above_zeros = np.copy(margin_value)
        is_above_zeros[is_above_zeros>=0] = True
        is_above_zeros[is_above_zeros<0] = False

        margin_label = margin_label * is_above_zeros
        
        return margin_label, margin_value

    def save_pt(self, save_path):
        results = {}
        results["categories_num"] = self.categories_num
        results["max_elements_num"] = self.max_elements_num
        results["data"] = self.data
        torch.save(results, save_path)

    def load_pt(self, load_path):
        results = torch.load(load_path)
        self.categories_num = results["categories_num"]
        self.max_elements_num = results["max_elements_num"]
        self.data = results["data"]

        self.copy_mode = self.size + 1 + self.categories_num
        self.margin_mode = self.copy_mode + 1
        self.generate_mode = self.margin_mode + 1
        self.no_obj_token = self.generate_mode + 1

    def get_all_element(self, p_dic, elements):
        if "children" in p_dic:
            for i in range(len(p_dic["children"])):
                cur_child = p_dic["children"][i]
                elements.append(cur_child)
                elements = self.get_all_element(cur_child, elements)
        return elements

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

    def __len__(self):
        return len(self.data)

    def render(self, layout):
        img = Image.new('RGB', (self.W, self.H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 13 * 13].reshape(-1, 13)
        box = layout[:, 1:].astype(np.float32)

        option_box = box.reshape(-1, 4, 3) # [box_num, xywh, [option, obj, value]]
        box_num = option_box.shape[0]
        obj_idx = option_box[:, :, 1] - self.no_obj_token
        is_generate = option_box[:, :, 0] == np.ones((box_num, 4))*self.generate_mode
        generate_value = is_generate * option_box[:, :, 2]
        is_copy = option_box[:, :, 0] == np.ones((box_num, 4))*self.copy_mode
        is_margin = option_box[:, :, 0] == np.ones((box_num, 4))*self.margin_mode

        result_box = option_box[0, :, 2][np.newaxis, :]
        for idx in range(1, len(layout)):
            cur_obj_idx = obj_idx[idx].astype(np.int32)
            cur_obj_idx[cur_obj_idx>len(result_box)] = 0
            cur_obj_idx[cur_obj_idx<0] = 0
            cur_copy_value = is_copy[idx] * result_box[-cur_obj_idx, range(4)]
            cur_bbox = generate_value[idx] + cur_copy_value
            margin_xy = result_box[-cur_obj_idx[:2], range(2)] + 0.5*result_box[-cur_obj_idx[:2], [2, 3]] + 0.5*cur_bbox[2:4] + option_box[idx, [0,1], 2]
            cur_margin_value = is_margin[idx] * np.concatenate((margin_xy, np.zeros(margin_xy.shape)), axis=-1)
            cur_bbox += cur_margin_value
            result_box = np.concatenate((result_box, cur_bbox[np.newaxis, :]), axis=0).astype(np.int32)

        # box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 255
        # box[:, [2, 3]] = box[:, [2, 3]] / self.size * 256
        # box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]
        result_box = result_box / (self.size - 1)

        result_box[:, [0, 2]] = result_box[:, [0, 2]] * self.W
        result_box[:, [1, 3]] = result_box[:, [1, 3]] * self.H
        # xywh to ltrb
        x1s = result_box[:, 0] - result_box[:, 2] / 2
        y1s = result_box[:, 1] - result_box[:, 3] / 2
        x2s = result_box[:, 0] + result_box[:, 2] / 2
        y2s = result_box[:, 1] + result_box[:, 3] / 2

        for i in range(len(layout)):
            # x1, y1, x2, y2 = box
            x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
            cat = layout[i][0]
            # col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            col = self.colors[int(cat-self.no_value_token-1)] if 0 <= cat-self.no_value_token-1 < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def render_normalized_layout(self, layout):
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 13 * 13].reshape(-1, 13)
        
        box = layout[:, 1:].astype(np.float32)
        option_box = box.reshape(-1, 4, 3) # [box_num, xywh, [option, obj, value]]
        box_num = option_box.shape[0]
        obj_idx = option_box[:, :, 1] - self.no_obj_token
        is_generate = option_box[:, :, 0] == np.ones((box_num, 4))*self.generate_mode
        generate_value = is_generate * option_box[:, :, 2]
        is_copy = option_box[:, :, 0] == np.ones((box_num, 4))*self.copy_mode
        is_margin = option_box[:, :, 0] == np.ones((box_num, 4))*self.margin_mode

        result_box = option_box[0, :, 2][np.newaxis, :]
        for idx in range(1, len(layout)):
            cur_obj_idx = obj_idx[idx].astype(np.int32)
            cur_obj_idx[cur_obj_idx>len(result_box)] = 0
            cur_obj_idx[cur_obj_idx<0] = 0
            cur_copy_value = is_copy[idx] * result_box[-cur_obj_idx, range(4)]
            cur_bbox = generate_value[idx] + cur_copy_value
            margin_xy = result_box[-cur_obj_idx[:2], range(2)] + 0.5*result_box[-cur_obj_idx[:2], [2, 3]] + 0.5*cur_bbox[2:4] + option_box[idx, [0,1], 2]
            cur_margin_value = is_margin[idx] * np.concatenate((margin_xy, np.zeros(margin_xy.shape)), axis=-1)
            cur_bbox += cur_margin_value
            result_box = np.concatenate((result_box, cur_bbox[np.newaxis, :]), axis=0).astype(np.int32)

        # box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 255
        # box[:, [2, 3]] = box[:, [2, 3]] / self.size * 256
        # box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]
        result_box = result_box / (self.size - 1)
        result_box = np.clip(result_box, 0, 1)
        label = layout[:, 0].astype(np.int32) - (self.size+1)
        # label[label>self.categories_num] = 0
        # label[label<0] = 0
        return (result_box, label)

    def calculate_prob(self, layout):
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 13 * 13].reshape(-1, 13)
        
        box = layout[:, 1:].astype(np.float32)
        option_box = box.reshape(-1, 4, 3) # [box_num, xywh, [option, obj, value]]
        box_num = option_box.shape[0]
        is_generate = option_box[:, :, 0] == np.ones((box_num, 4))*self.generate_mode
        is_copy = option_box[:, :, 0] == np.ones((box_num, 4))*self.copy_mode
        is_margin = option_box[:, :2, 0] == np.ones((box_num, 2))*self.margin_mode

        generate_chosen = np.sum(is_generate)
        copy_chosen = np.sum(is_copy)
        margin_chosen = np.sum(is_margin)

        return box_num, generate_chosen, copy_chosen, margin_chosen

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        return layout['x'], layout['y']    