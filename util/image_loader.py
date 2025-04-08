import os
import random
import numpy as np
from PIL import Image

def load_image(image_path,unit_n_H:int,unit_n_W:int,unit_size_H:int,unit_size_W:int,stride_H:int,stride_W:int,
               target_size=None,patch_on=True,seq_on=False):
    # get list of images file name in the folder
    imgs = os.listdir(image_path)

    # seq_on for video to load sequence of frames
    if seq_on:
        imgs = [imgname for imgname in imgs if imgname[-4:] == '.jpg']
        for imgname in imgs:
            if imgname[-4:] != '.jpg':
                continue
            # open image file
            img = Image.open(image_path + imgname).convert("RGB")
            # divide to patches
            img_ith = int(imgname.split('.')[0])
            img_seq=[Image.open(image_path + str(img_ith-i).zfill(4)+'.jpg').convert("RGB")
                    for i in range(0,min(25,img_ith),5)]
            
            patches = []
            if patch_on:
                for h in range(unit_n_H):
                    patches.append([])
                    for w in range(unit_n_W):
                        patch = img.crop((w*stride_W, h*stride_H, w*stride_W+unit_size_W, h*stride_H+unit_size_H))
                        if target_size is not None:
                            patch = patch.resize(target_size,resample=Image.BICUBIC)
                        # patch.show()
                        patches[h].append(patch)
            # img = img.resize((,H),resample=Image.NEAREST)
            if target_size is not None:
                # img = img.resize(target_size,resample=Image.NEAREST)
                for i in range(len(img_seq)):
                    img_seq[i] = img_seq[i].resize(target_size,resample=Image.BICUBIC)
            # img.show()
            yield img_seq,patches,imgname
    else:
        for imgname in imgs:
            if not (imgname[-4:] == '.jpg' or imgname[-4:] == '.png'):
                continue
            # open image file
            img = Image.open(image_path + imgname).convert("RGB")

            # get image size
            ori_size = img.size
            # divide to patches
            
            patches = []
            if patch_on:
                for h in range(unit_n_H):
                    patches.append([])
                    for w in range(unit_n_W):
                        patch = img.crop((w*stride_W, h*stride_H, w*stride_W+unit_size_W, h*stride_H+unit_size_H))
                        if target_size is not None:
                            patch = patch.resize(target_size,resample=Image.BICUBIC)
                        # patch.show()
                        patches[h].append(patch)
            # img = img.resize((,H),resample=Image.NEAREST)
            if target_size is not None:
                img = img.resize(target_size,resample=Image.BICUBIC)
            # img.show()
            yield img,patches,imgname,ori_size

def load_image_pad(image_path,patch_div,stride_div,
               target_size=None,patch_on=True,seq_on=False,sample_on=False):
    # get list of images file name in the folder
    imgs = os.listdir(image_path)
    unit_n = (patch_div+2)*stride_div-stride_div-1
    # seq_on for video to load sequence of frames
    if seq_on:
        raise NotImplementedError
    else:
        sample_count = 0
        for imgname in imgs:
            if sample_on:
                if 'DHF1K' in image_path:
                    if int(imgname.split('.')[0]) % 300 != 0:
                        continue
                elif 'ucf' in image_path:
                    sample_count += 1
                    if sample_count % 10 !=0:
                        continue
                elif 'Hollywood2' in image_path:
                    if int(imgname.split('.')[0].split('_')[-1]) % 300 != 0:
                        continue
                elif 'SUBWAY' in image_path:
                    pass
                elif 'THUE' in image_path:
                    if int(imgname.split('.')[0]) % 300 != 0:
                        continue
                elif 'DREYEVE' in image_path:
                    if int(imgname.split('.')[0]) % 300 != 0:
                        continue
                else:
                    raise NotImplementedError
                
            if not (imgname[-4:] == '.jpg' or imgname[-4:] == '.png'):
                continue
            # open image file
            img = Image.open(image_path + imgname).convert("RGB")

            # get image size
            ori_size = img.size
            if target_size is not None:
                # get the size of unit and stride
                scale_size=(target_size[0]*patch_div,target_size[1]*patch_div)
                unit_size_W = scale_size[0]//patch_div
                unit_size_H = scale_size[1]//patch_div
                stride_W = scale_size[0]//patch_div//stride_div
                stride_H = scale_size[1]//patch_div//stride_div

                #padding with 0
                img = img.resize(scale_size,resample=Image.BICUBIC)
                scale_size=(target_size[0]*(patch_div+2)-target_size[0]//stride_div*2,target_size[1]*(patch_div+2)-target_size[1]//stride_div*2)
                
                img_edge = Image.new('RGB',scale_size,color=0)
                img_edge.paste(img,(target_size[0]-target_size[0]//stride_div,target_size[1]-target_size[1]//stride_div))
                # img_edge.paste(img,(1,1))
                img=img_edge
            # divide to patches
            
            patches = []
            if patch_on:
                for h in range(unit_n):
                    patches.append([])
                    for w in range(unit_n):
                        patch = img.crop((w*stride_W, h*stride_H, w*stride_W+unit_size_W, h*stride_H+unit_size_H))
                        if target_size is not None:
                            patch = patch.resize(target_size,resample=Image.BICUBIC)
                        patches[h].append(patch)
            # img = img.resize((,H),resample=Image.NEAREST)
            if target_size is not None:
                img = img.resize(target_size,resample=Image.BICUBIC)
            # img.show()
            yield img,patches,imgname,ori_size

def load_image_div(image_path,patch_div,stride_div,
               target_size=None,patch_on=True,seq_on=False):
    # get list of images file name in the folder
    imgs = os.listdir(image_path)
    unit_n = patch_div*stride_div-stride_div+1
    # seq_on for video to load sequence of frames
    if seq_on:
        raise NotImplementedError
    else:
        for imgname in imgs:
            if not (imgname[-4:] == '.jpg' or imgname[-4:] == '.png'):
                continue
            # open image file
            img = Image.open(image_path + imgname).convert("RGB")

            # get image size
            ori_size = img.size
            scale_size=(target_size[0]*patch_div,target_size[1]*patch_div)
            if target_size is not None:
                img = img.resize(scale_size,resample=Image.BICUBIC)
                unit_size_W = scale_size[0]//patch_div
                unit_size_H = scale_size[1]//patch_div
                stride_W = scale_size[0]//patch_div//stride_div
                stride_H = scale_size[1]//patch_div//stride_div
            # divide to patches
            
            patches = []
            if patch_on:
                for h in range(unit_n):
                    patches.append([])
                    for w in range(unit_n):
                        patch = img.crop((w*stride_W, h*stride_H, w*stride_W+unit_size_W, h*stride_H+unit_size_H))
                        if target_size is not None:
                            patch = patch.resize(target_size,resample=Image.BICUBIC)
                        # patch.show()
                        patches[h].append(patch)
            # img = img.resize((,H),resample=Image.NEAREST)
            if target_size is not None:
                img = img.resize(target_size,resample=Image.BICUBIC)
            # img.show()
            yield img,patches,imgname,ori_size

def other_maps(fix_path):
    folders = os.listdir(fix_path)
    for i in range(10):
        i_folder = random.randint(0,len(folders)-1)
        fix_maps = os.listdir(os.path.join(fix_path,folders[i_folder],'fixation'))
        # delete maps not end with .png or .jpg
        fix_maps = [fix_map for fix_map in fix_maps if fix_map[-4:] == '.png' or fix_map[-4:] == '.jpg']
        i_img = random.randint(0,len(fix_maps)-1)
        # fix_map = Image.open(os.path.join(fix_path,folders[i_folder],os.listdir(os.path.join(fix_path,folders[i_folder]))[i_img])).convert('L')
        # fix_map = sio.loadmat(os.path.join(fix_path,str(i_folder).zfill(4),'fixation',os.listdir(os.path.join(fix_path,str(i_folder).zfill(4),'fixation','maps'))[i_img][:-4]+'.mat'))['I']
        fix_map = Image.open(os.path.join(fix_path,folders[i_folder],'fixation',fix_maps[i_img])).convert('L')
        fix_map = np.array(fix_map).astype(np.float32)/255
        if i == 0:
            # fix_map = np.array(fix_map).astype(np.float32)
            fix_map = fix_map>0
            maps=fix_map
        else:
            # fix_map = np.array(fix_map).astype(np.float32)
            fix_map = fix_map>0
            maps=maps+fix_map
    maps = np.clip(maps,0,1)
    return maps

def other_maps_scale(fix_path,ori_size,sample_size=10):
    folders = os.listdir(fix_path)
    # for folder in folders:
    #     if '.' in folder or 'py' in folder:
    #         # remove folder with '.' in the name
    #         folders.remove(folder)
    for i in range(sample_size):
        i_folder = random.randint(0,len(folders)-1)
        fix_maps = os.listdir(os.path.join(fix_path,folders[i_folder],'fixation'))
        # delete maps not end with .png or .jpg
        fix_maps = [fix_map for fix_map in fix_maps if fix_map[-4:] == '.png' or fix_map[-4:] == '.jpg']
        i_img = random.randint(0,len(fix_maps)-1)
        # fix_map = Image.open(os.path.join(fix_path,folders[i_folder],os.listdir(os.path.join(fix_path,folders[i_folder]))[i_img])).convert('L')
        # fix_map = sio.loadmat(os.path.join(fix_path,str(i_folder).zfill(4),'fixation',os.listdir(os.path.join(fix_path,str(i_folder).zfill(4),'fixation','maps'))[i_img][:-4]+'.mat'))['I']
        fix_map = Image.open(os.path.join(fix_path,folders[i_folder],'fixation',fix_maps[i_img])).convert('L')
        fix_map = fix_map.resize(ori_size,resample=Image.BICUBIC)
        fix_map = np.array(fix_map).astype(np.float32)/255
        if i == 0:
            # fix_map = np.array(fix_map).astype(np.float32)
            fix_map = fix_map>(np.max(fix_map)*0.4)
            maps=fix_map
        else:
            # fix_map = np.array(fix_map).astype(np.float32)
            fix_map = fix_map>(np.max(fix_map)*0.4)
            maps=maps+fix_map
    maps = np.clip(maps,0,1)
    # print(np.sum(maps))
    return maps