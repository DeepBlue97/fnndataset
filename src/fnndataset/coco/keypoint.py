from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data.dataset import Dataset


import random
import cv2
import albumentations as A
import json
import platform
import math


def label2reg(keypoints, cx, cy, img_size):
    # cx = int(center[0]*img_size/4)
    # cy = int(center[1]*img_size/4)
    #print("cx cy: ", cx, cy)
    heatmaps = np.zeros((len(keypoints)//3*2, img_size//4, img_size//4), dtype=np.float32)
    # print(keypoints)
    for i in range(len(keypoints)//3):
        if keypoints[i*3+2]==0:
            continue

        x = keypoints[i*3]*img_size//4
        y = keypoints[i*3+1]*img_size//4
        if x==img_size//4:x=(img_size//4-1)
        if y==img_size//4:y=(img_size//4-1)
        if x>img_size//4 or x<0 or y>img_size//4 or y<0:
            continue

        reg_x = x-cx
        reg_y = y-cy
        # print(reg_x,reg_y)
        # heatmaps[i*2][cy][cx] = reg_x#/(img_size//4)
        # heatmaps[i*2+1][cy][cx] = reg_y#/(img_size//4)



        for j in range(cy-2,cy+3):
            if j<0 or j>img_size//4-1:
                continue
            for k in range(cx-2,cx+3): 
                if k<0 or k>img_size//4-1:
                    continue
                if cx<img_size//4/2-1:
                    heatmaps[i*2][j][k] = reg_x-(cx-k)#/(img_size//4)
                else:
                    heatmaps[i*2][j][k] = reg_x+(cx-k)#/(img_size//4)
                if cy<img_size//4/2-1:
                    heatmaps[i*2+1][j][k] = reg_y-(cy-j)#/(img_size//4)
                else:
                    heatmaps[i*2+1][j][k] = reg_y+(cy-j)
  
    return heatmaps


def label2offset(keypoints, cx, cy, regs, img_size):
    heatmaps = np.zeros((len(keypoints)//3*2, img_size//4, img_size//4), dtype=np.float32)
    # print(keypoints)
    #print(regs.shape)#(14, 48, 48)
    for i in range(len(keypoints)//3):
        if keypoints[i*3+2]==0:
            continue

        large_x = int(keypoints[i*3]*img_size)
        large_y = int(keypoints[i*3+1]*img_size)


        small_x = int(regs[i*2,cy,cx]+cx)
        small_y = int(regs[i*2+1,cy,cx]+cy)

        
        offset_x = large_x/4-small_x
        offset_y = large_y/4-small_y

        if small_x==img_size//4:small_x=(img_size//4-1)
        if small_y==img_size//4:small_y=(img_size//4-1)
        if small_x>img_size//4 or small_x<0 or small_y>img_size//4 or small_y<0:
            continue
        # print(offset_x, offset_y)
        
        # print()
        heatmaps[i*2][small_y][small_x] = offset_x#/(img_size//4)
        heatmaps[i*2+1][small_y][small_x] = offset_y#/(img_size//4)

    # print(heatmaps.shape)
    
    return heatmaps


def generate_heatmap(x, y, other_keypoints, size, sigma):
    #x,y  abs postion
    #other_keypoints   positive position
    sigma+=6
    heatmap = np.zeros(size)
    if x<0 or y<0 or x>=size[0] or y>=size[1]:
        return heatmap
    
    tops = [[x,y]]
    if len(other_keypoints)>0:
        #add other people's keypoints
        for i in range(len(other_keypoints)):
            x = int(other_keypoints[i][0]*size[0])
            y = int(other_keypoints[i][1]*size[1])
            if x==size[0]:x=(size[0]-1)
            if y==size[1]:y=(size[1]-1)
            if x>size[0] or x<0 or  y>size[1] or y<0: continue
            tops.append([x,y])


    for top in tops:
        #heatmap[top[1]][top[0]] = 1
        x,y = top
        x0 = max(0,x-sigma//2)
        x1 = min(size[0],x+sigma//2)
        y0 = max(0,y-sigma//2)
        y1 = min(size[1],y+sigma//2)


        for map_y in range(y0, y1):
            for map_x in range(x0, x1):
                d2 = ((map_x  - x) ** 2 + (map_y  - y) ** 2)**0.5

                if d2<=sigma//2:
                    heatmap[map_y, map_x] += math.exp(-d2/(sigma//2)*3)
                    # heatmap[map_y, map_x] += math.exp(-d2/sigma**2)
                #print(keypoint_map[map_y, map_x])
                if heatmap[map_y, map_x] > 1:
                    #不同关键点可能重合，这里累加
                    heatmap[map_y, map_x] = 1

    # heatmap[heatmap<0.1] = 0
    return heatmap


def label2heatmap(keypoints, other_keypoints, img_size):
    #keypoints: target person
    #other_keypoints: other people's keypoints need to be add to the heatmap
    heatmaps = []
    # print(keypoints)

    keypoints_range = np.reshape(keypoints,(-1,3))
    keypoints_range = keypoints_range[keypoints_range[:,2]>0]
    # print(keypoints_range)
    min_x = np.min(keypoints_range[:,0])
    min_y = np.min(keypoints_range[:,1])
    max_x = np.max(keypoints_range[:,0])
    max_y = np.max(keypoints_range[:,1])
    area = (max_y-min_y)*(max_x-min_x)
    sigma = 3
    if area < 0.16:
        sigma = 3
    elif area < 0.3:
        sigma = 5
    else:
        sigma = 7
    

    for i in range(0,len(keypoints),3):
        if keypoints[i+2]==0:
            heatmaps.append(np.zeros((img_size//4, img_size//4)))
            continue

        x = int(keypoints[i]*img_size//4) #取值应该是0-47
        y = int(keypoints[i+1]*img_size//4)
        if x==img_size//4:x=(img_size//4-1)
        if y==img_size//4:y=(img_size//4-1)
        if x>img_size//4 or x<0:x=-1
        if y>img_size//4 or y<0:y=-1
        heatmap = generate_heatmap(x, y, other_keypoints[i//3], (img_size//4, img_size//4),sigma)

        heatmaps.append(heatmap)

    heatmaps = np.array(heatmaps, dtype=np.float32)

    #heatmaps_bg = np.reshape(1 - heatmaps.max(axis=0), (1,heatmaps.shape[1],heatmaps.shape[2]))
    # print(heatmaps.shape, heatmaps_bg.shape)

    #heatmaps = np.concatenate([heatmaps,heatmaps_bg], axis=0)

    
    return heatmaps,sigma


def label2center(cx, cy, other_centers, img_size, sigma):
    heatmaps = []
    # print(label)

    # cx = int(center[0]*img_size/4)
    # cy = int(center[1]*img_size/4)
    
    heatmap = generate_heatmap(cx, cy, other_centers, (img_size//4, img_size//4),sigma+2)
    heatmaps.append(heatmap)

    heatmaps = np.array(heatmaps, dtype=np.float32)
    # print(heatmaps.shape)
    
    return heatmaps


class TensorDataset(Dataset):

    def __init__(self, label_path, img_dir, img_size, data_aug=None):

        with open(label_path,'r') as f:
            data_labels = json.loads(f.readlines()[0])  
            # random.shuffle(label_list)

        self.data_labels = data_labels
        self.img_dir = img_dir
        self.data_aug = data_aug
        self.img_size = img_size


        self.interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, 
                                cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]

    def __getitem__(self, index):

        item = self.data_labels[index]
        # print(len(self.data_labels), index)
        # while 'yoga_img_514' not in item["img_name"]:
        #     index+=1
        #     item = self.data_labels[index]

        # while len(item['other_centers'])==0:
        #     index+=1
        #     item = self.data_labels[index]
        # print("----")
        # if '000000103797_0' in item["img_name"]:
        #     print(item)
        """
        item = {
                     "img_name":save_name,
                     "keypoints":save_keypoints,
                     "center":save_center,
                     "other_centers":other_centers,
                     "other_keypoints":other_keypoints,
                    }
        """
        # label_str_list = label_str.strip().split(',')
        # [name,h,w,keypoints...]
        img_path = os.path.join(self.img_dir, item["img_name"])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.img_size, self.img_size),
                        interpolation=random.choice(self.interp_methods))


        #### Data Augmentation
        # print(item)
        if self.data_aug is not None:
            img, item = self.data_aug(img, item)
        # print(item)
        # print()

        # cv2.imwrite(os.path.join("img.jpg"), img)

        img = img.astype(np.float32)
        img = np.transpose(img,axes=[2,0,1])


        keypoints = item["keypoints"]
        center = item['center']
        other_centers = item["other_centers"]
        other_keypoints = item["other_keypoints"]

        #print(keypoints)
        #[0.640625   0.7760417  2, ] (21,)

        kps_mask = np.ones(len(keypoints)//3)
        for i in range(len(keypoints)//3):
            ##0没有标注;1有标注不可见（被遮挡）;2有标注可见
            if keypoints[i*3+2]==0:
                kps_mask[i] = 0


        # img = img.transpose((1,2,0))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join("_img.jpg"), img)


        heatmaps,sigma = label2heatmap(keypoints, other_keypoints, self.img_size) #(17, 48, 48)
        #超出边界则设为全0


        # img = img.transpose((1,2,0))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # hm = cv2.resize(np.sum(heatmaps,axis=0)*255,(192,192))
        # cv2.imwrite(os.path.join("_hm.jpg"), hm)
        # cv2.imwrite(os.path.join("_img.jpg"), img)


        cx = min(max(0,int(center[0]*self.img_size//4)),self.img_size//4-1)
        cy = min(max(0,int(center[1]*self.img_size//4)),self.img_size//4-1)
        # if '000000103797_0' in item["img_name"]:
        #     print("---data_tools 404 cx,cy: ",cx,cy, center)

        centers = label2center(cx, cy, other_centers, self.img_size, sigma) #(1, 48, 48)
        # cv2.imwrite(os.path.join("_img.jpg"), centers[0]*255)
        #print(centers[0,21:26,21:26])
        # print(centers[0,12:20,27:34])
        # print(centers[0,y,x],x,y)
        
        
        # cx2,cy2 = extract_keypoints(centers)


        # cx2,cy2 = maxPoint(centers)
        # cx2,cy2 = cx2[0][0],cy2[0][0]
        # print(cx2,cy2)
        # if cx!=cx2 or cy!=cy2:
        #     # cv2.imwrite(os.path.join("_img.jpg"), centers[0]*255)
        # print(centers[0,17:21,22:26])
        #     print(cx,cy ,cx2,cy2)
        #     raise Exception("center changed after label2center!")
        
        # print(keypoints[0]*48,keypoints[1]*48)
        regs = label2reg(keypoints, cx, cy, self.img_size) #(14, 48, 48)
        # cv2.imwrite(os.path.join("_regs.jpg"), regs[0]*255)
        # print(regs[0][22:26,22:26])
        # print(regs[1][22:26,22:26])

        # print("regs[0,cy,cx]: ", regs[0,cy,cx])
        # for i in range(14):
        #     print(regs[i,y,x])

        offsets = label2offset(keypoints, cx, cy, regs, self.img_size)#(14, 48, 48)
        # for i in range(14):
        #     print(regs[i,y,x])

        # print(heatmaps.shape, regs.shape, offsets.shape)


        # img = img.transpose((1,2,0))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # for i in range(7):
        #     cv2.imwrite("_heatmaps%d.jpg" % i,cv2.resize(heatmaps[i]*255-i*20,(192,192)))
        #     img[:,:,0]+=cv2.resize(heatmaps[i]*255-i*20,(192,192))
        # cv2.imwrite(os.path.join("_img.jpg"), img)



        labels = np.concatenate([heatmaps,centers,regs,offsets],axis=0)
        # print(heatmaps.shape,centers.shape,regs.shape,offsets.shape,labels.shape)
        # print(labels.shape)
        # return img, [labels, kps_mask, img_path]
        return img, (labels, kps_mask, img_path)

    def __len__(self):
        return len(self.data_labels)
