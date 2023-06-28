from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math


def get_cwh(box):
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    return cx, cy, w, h


def convert_xywh_to_xyxy(box):
    # x_min, y_min, w, h = box
    # x_max = x_min + w
    # y_max = y_min + h
    xc, yc, w, h = box
    x_min = xc - w*0.5
    y_min = yc - h*0.5
    x_max = xc + w*0.5
    y_max = yc + h*0.5
    
#    if w <1:
#        w=1
#    if h <1:
#        h=1    
    return([x_min, y_min, x_max, y_max])


def cal_geometry_feats(boxes, NumFeats = 8, Directed = True):
    num_boxes = boxes.shape[0]
    
    w, h = 256, 256
    # w = np.max([box[2] for box in boxes])
    # h = np.max([box[3] for box in boxes])
    # print(w, h)

    scale = w * h
    diag_len = math.sqrt(w ** 2 + h ** 2)
    
    feats = np.zeros([num_boxes, num_boxes, NumFeats], dtype='float')
    
    for i in range(num_boxes):
        if Directed:
            start = 0
        else:
            start = i
        
        for j in range(start, num_boxes):          
            boxc1, boxc2 = boxes[i], boxes[j]

            cx1, cy1, w1, h1 = boxc1
            cx2, cy2, w2, h2 = boxc2  
            
            #Convet to xyxy format, as it is saved as xywh
            box1 = convert_xywh_to_xyxy(boxc1)
            box2 = convert_xywh_to_xyxy(boxc2)

            # cx1, cy1, w1, h1 = get_cwh(box1)
            # cx2, cy2, w2, h2 = get_cwh(box2)
            
            x_min1, y_min1, x_max1, y_max1 = box1
            x_min2, y_min2, x_max2, y_max2 = box2
            
            # scale
            scale1 = w1 * h1
            scale2 = w2 * h2
            
            # Offset
            offsetx = cx2 - cx1
            offsety = cy2 - cy1
            
            # Aspect ratio
            aspect1 = w1 / h1
            aspect2 = w2 / h2
            
            # Overlap (IoU)
            i_xmin = max(x_min1, x_min2)
            i_ymin = max(y_min1, y_min2)
            i_xmax = min(x_max1, x_max2)
            i_ymax = min(y_max1, y_max2)
            iw = max(i_xmax - i_xmin + 1, 0)
            ih = max(i_ymax - i_ymin + 1, 0)
            areaI = iw * ih
            areaU = scale1 + scale2 - areaI
            
            # dist
            dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            
            # angle
            angle = math.atan2(cy2 - cy1, cx2 - cx1)

            f1 = offsetx / math.sqrt(scale1)
            f2 = offsety / math.sqrt(scale1)  
 
            f3 = math.sqrt(scale2 / scale1)                
            f4 = areaI / areaU
            f5 = aspect1
            f6 = aspect2
            f7 = dist / diag_len
            f8 = angle
            feat = [f1, f2, f3, f4, f5, f6, f7, f8]
            feats[i][j] = np.array(feat)
    
    # no_nan_t = np.where(np.isnan(feats) == True)[0]

    return feats

def build_geometry_graph(feats, Directed=True):
    num_boxes = feats.shape[0]
    edges = []
    relas = []
    
    #if id == '71073':
        #print('breakpoint')
    
    for i in range(num_boxes):
        if Directed:
            start = 0
        else:
            start = i
        for j in range(start, num_boxes):
            if i==j:
                continue
            # iou and dist thresholds
#            if feats[i][j][3] < Iou or feats[i][j][6] > Dist:
#                continue
            edges.append([i, j])
            relas.append(feats[i][j])

    
    # in case some trouble is met
    if edges == []:
#        print('id: ', id )
        # f = open("../data/single_component_images_directed.txt", "a")
        # f.write('%s\n'%(id))
#        edges.append([0, 1])
#        relas.append(feats[0][1])
        edges.append([0, 0])
        relas.append(feats[0][0]) 

    edges = np.array(edges)
    relas = np.array(relas)
    graph = {}
    graph['edges'] = edges
    graph['feats'] = relas
    return graph