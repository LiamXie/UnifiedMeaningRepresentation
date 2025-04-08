import torch
import numpy as np
import os
from PIL import Image

from util import utils

def split_train_val(dataset,dataset_path,mode):
    if dataset=='THUE_FREE':
        # THUE_FREE has 104 folders, 70 for training, 34 for validation

        # get the absolute path of all folders
        folders = os.listdir(dataset_path)
        for i in range(len(folders)):
            folders[i] = os.path.join(dataset_path,folders[i])
        folders.sort()
        if mode == 'train':
            train_folders = []
            val_folders = []
            for folder in folders:
                if int(os.path.basename(folder))>44:
                    train_folders.append(folder)
                else:
                    val_folders.append(folder)
        if mode == 'test' or mode == 'val':
            train_folders = []
            val_folders = folders
        return train_folders,val_folders
    elif dataset=='THUE_TASK':
        folders = os.listdir(dataset_path)
        for i in range(len(folders)):
            folders[i] = os.path.join(dataset_path,folders[i])
        folders.sort()
        train_folders = []
        val_folders = []
        if mode=='train' or mode=='val':
            for folder in folders:
                if int(os.path.basename(folder))>44:
                    train_folders.append(folder)
                else:
                    val_folders.append(folder)
        if mode=='test':
            for folder in folders:
                if int(os.path.basename(folder))>44:
                    train_folders.append(folder)
                else:
                    val_folders.append(folder)
        return train_folders,val_folders
    elif dataset=='UCF':
        train_folders = os.listdir(os.path.join(dataset_path,'training'))
        for i in range(len(train_folders)):
            train_folders[i] = os.path.join(dataset_path,'training',train_folders[i])
        val_folders = os.listdir(os.path.join(dataset_path,'testing'))
        for i in range(len(val_folders)):
            val_folders[i] = os.path.join(dataset_path,'testing',val_folders[i])
        return train_folders,val_folders
    elif dataset=='DHF1K':
        # DHF1K has 700 folders, 450 for training, 250 for validation
        folders = os.listdir(dataset_path)
        for i in range(len(folders)):
            folders[i] = os.path.join(dataset_path,folders[i])
        folders.sort()
        train_folders = folders[:450]
        val_folders = folders[450:]
        return train_folders,val_folders
    elif dataset=='Hollywood2':
        train_folders = os.listdir(os.path.join(dataset_path,'training'))
        for i in range(len(train_folders)):
            train_folders[i] = os.path.join(dataset_path,'training',train_folders[i])
        val_folders = os.listdir(os.path.join(dataset_path,'testing'))
        for i in range(len(val_folders)):
            val_folders[i] = os.path.join(dataset_path,'testing',val_folders[i])
        return train_folders,val_folders
    
def scale_fixation(fixation_map,target_size):
    fix = np.array(fixation_map).astype(np.float32)/255
    fix = fix>0
    if fix.max() == 0:
        raise ValueError('fixation map is all zero')
    # get all fixation points and scale to target size
    fix_points = np.where(fix)
    fix_points = np.array(fix_points).T
    fix_points = fix_points * np.array(target_size)/np.array(fix.shape)
    fix_points = fix_points.astype(int)
    fix = np.zeros(target_size,dtype=np.float32)
    for point in fix_points:
        fix[point[0],point[1]] = 1
    return fix
    
# class of data loader, inherit from torch.utils.data.Dataset, reading video frame from folder
class COMPDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 datasets:dict,
                mode,
                preprocess,
                num_frames=16,
                frame_modulo=5, # select frames every num_steps frames
                # norm setting
                mean=(0.45, 0.45, 0.45),
                std=(0.225, 0.225, 0.225),
                target_size=None,
                left_right=False
    ):
        self.datasets = datasets
        self.frame_modulo=frame_modulo
        self.mode = mode
        self.num_frames = num_frames
        self._mean = mean
        self._std = std
        self.target_size=target_size
        self.preprocess = preprocess

        self.video_list=[]
        self.folder_images_list=[]
        if mode == "train":
            # get key and value of datasets
            for key, value in datasets.items():
                train_folders,_ = split_train_val(key,value,mode)
                self.video_list=self.video_list+train_folders
        if mode == "val" or mode == "test":
            for key, value in datasets.items():
                _,val_folders = split_train_val(key,value,mode)
                self.video_list=self.video_list+val_folders
        
        for video in self.video_list:
            if 'THUE' in video or 'UCF' in video:
                for img in os.listdir(os.path.join(video,'images')):
                    if img[-4:] != '.jpg' and img[-4:] != '.png':
                        continue
                    self.folder_images_list.append([video,img])
            else:
                for ith, img in enumerate(os.listdir(os.path.join(video,'images'))):
                    if img[-4:] != '.jpg' and img[-4:] != '.png':
                        continue
                    if 'Hollywood2' in video:
                        if int(img.split('.')[0].split('_')[-1]) % 30 != 0:
                            continue
                    if 'DHF1K' in video:
                        if int(img.split('.')[0]) % 30 != 0:
                            continue
                    self.folder_images_list.append([video,img])

    def __len__(self):
        return len(self.folder_images_list)

    def __getitem__(self, idx):
        '''
        Given the video_image index, return the frame, saliency map and fixation map.
        Args:
            idx (int): the video index. 
        Returns:
            frame (tensor): [C, H, W]
            sal (tensor): [H, W] 
            fix (tensor): [H, W]
        '''
        frame = Image.open(os.path.join(self.folder_images_list[idx][0],'images',self.folder_images_list[idx][1])).convert("RGB")
        frame = frame.resize(self.target_size,resample=Image.CUBIC)
        # frame = self.preprocess(frame,return_tensors="pt")
        # frame = frame['pixel_values'].squeeze(0)
        frame = self. preprocess(frame)
        

        # load saliency map
        if os.path.exists(os.path.join(self.folder_images_list[idx][0],'maps',self.folder_images_list[idx][1])):
            sal = Image.open(os.path.join(self.folder_images_list[idx][0],'maps',self.folder_images_list[idx][1])).convert('L')
        else:
            # replace jpg with png or png with jpg
            if self.folder_images_list[idx][1][-4:] == '.png':
                sal = Image.open(os.path.join(self.folder_images_list[idx][0],'maps',self.folder_images_list[idx][1].replace('.png','.jpg'))).convert('L')
            else:
                sal = Image.open(os.path.join(self.folder_images_list[idx][0],'maps',self.folder_images_list[idx][1].replace('.jpg','.png'))).convert('L')
        sal = sal.resize(self.target_size,resample=Image.CUBIC)
        sal = np.array(sal).astype(np.float32)/255
        sal = torch.tensor(sal)
        sal = utils.prob_tensor(sal)

        # load fixation map
        if self.mode =='train' or self.mode == 'val':
            
            if os.path.exists(os.path.join(self.folder_images_list[idx][0],'fixation',self.folder_images_list[idx][1])):
                fix = Image.open(os.path.join(self.folder_images_list[idx][0],'fixation',self.folder_images_list[idx][1])).convert('L')

            else:
                # replace jpg with png or png with jpg
                if self.folder_images_list[idx][1][-4:] == '.png':
                    fix = Image.open(os.path.join(self.folder_images_list[idx][0],'fixation',self.folder_images_list[idx][1].replace('.png','.jpg'))).convert('L')
                else:
                    fix = Image.open(os.path.join(self.folder_images_list[idx][0],'fixation',self.folder_images_list[idx][1].replace('.jpg','.png'))).convert('L')
            # resize
            fix = scale_fixation(fix,self.target_size)
            fix = torch.tensor(fix)
            fix=torch.gt(fix, 0.4)
        
        # load original fixation map for test
        elif self.mode == 'test':
            if os.path.exists(os.path.join(self.folder_images_list[idx][0],'fixation',self.folder_images_list[idx][1])):
                fix = Image.open(os.path.join(self.folder_images_list[idx][0],'fixation',self.folder_images_list[idx][1])).convert('L')

            else:
                # replace jpg with png or png with jpg
                if self.folder_images_list[idx][1][-4:] == '.png':
                    fix = Image.open(os.path.join(self.folder_images_list[idx][0],'fixation',self.folder_images_list[idx][1].replace('.png','.jpg'))).convert('L')
                else:
                    fix = Image.open(os.path.join(self.folder_images_list[idx][0],'fixation',self.folder_images_list[idx][1].replace('.jpg','.png'))).convert('L')
            fix = np.array(fix).astype(np.float32)/255
            fix = fix / fix.max()
            fix = torch.tensor(fix)
            fix=torch.gt(fix, 0.4)

        return frame, sal, fix