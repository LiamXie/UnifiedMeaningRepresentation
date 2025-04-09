# Calculating meaningmap for different datasets
from util.image_loader import load_image_pad
import open_clip
import torch
import numpy as np
from PIL import Image
import os
import cv2
import datetime
# from transformers import ViTModel, ViTImageProcessor
import json

MODEL_NAMES=['laion/CLIP-ViT-B-16-laion2B-s34B-b88K']
DATASETS = ['THUE_FREE']
# DATASETS = ['DHF1K','UCF','Hollywood2','LEDOV']
SAMPLE_ON=False

# define a feature computation function
def compute_feature(img,net,preprocess,device,model_name):
    if model_name == 'vit_base_patch16_224.mae':
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = net(img)

    elif model_name == 'google/vit-base-patch16-224-in21k':
        img = preprocess(img,return_tensors="pt").to(device)#.unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = net(**img).pooler_output#.last_hidden_state
    
    else:
        img_clip = preprocess(img).unsqueeze(0).to(device)
        # get image features
        with torch.no_grad():
            image_features = net.encode_image(img_clip)
    return image_features

def compute_meaning_map(patches_sims,logit_scale,ori_size):
    patches_sims = np.array(patches_sims)
    # remove dimension with size 1
    patches_sims = np.squeeze(patches_sims)

    meaning_map=patches_sims
    # softmax
    meaning_map = np.exp(meaning_map * logit_scale)
    meaning_map = cv2.resize(meaning_map,ori_size,interpolation=cv2.INTER_CUBIC)
    meaning_map = meaning_map / np.max(meaning_map)
    return meaning_map

def load_model(model_name):
    net, preprocess = open_clip.create_model_from_pretrained('hf-hub:'+model_name)
    return net, preprocess

def compute_mean_feature(net,preprocess,device,target_size=None,model_name=None):
    with torch.no_grad():
        feat_num=1000 # number of random features 
        mean_feature = None
        for i in range(feat_num):
            if target_size is not None:
                img = np.random.randint(0,256,(3,target_size[0],target_size[1]),dtype=np.uint8)
            else:
                raise NotImplementedError
            img = Image.fromarray(img.transpose(1,2,0))
            image_features = compute_feature(img,net,preprocess,device,model_name)
            if mean_feature is None:
                mean_feature = image_features
            else:
                mean_feature += image_features
        
        mean_feature /= feat_num

        return mean_feature

def load_tokenizer(name):
    tokenizer = open_clip.get_tokenizer('hf-hub:'+name)
    return tokenizer

if __name__ == '__main__':
    for DATASETS in DATASETS:
        for MODEL_NAME in MODEL_NAMES:
            # load dataset_info
            with open('./dataset_info.json','r') as f:
                datasets_info = json.load(f)

            # get parameters
            DIV_PATCH_LIST=datasets_info[DATASETS]['DIV_PATCH_LIST']
            DIV_STRIDE=datasets_info[DATASETS]['DIV_STRIDE']
            dataset_path=datasets_info[DATASETS]['DATASET_PATH']
            
            target_size = (224,224)
            
            # load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            net, preprocess = load_model(MODEL_NAME)
            tokenizer = load_tokenizer(MODEL_NAME)
            net.to(device).eval()

            # get logit_scale
            logit_scale = net.logit_scale.item()

            # create save folders
            for DIV_PATCH in DIV_PATCH_LIST:
                save_path_free = './outputs/'+DATASETS+'/FREE/'+MODEL_NAME+'_'+str(DIV_PATCH)+'_'+str(DIV_STRIDE)
                if not os.path.exists(save_path_free):
                    os.makedirs(save_path_free)
            save_path_final_free = './outputs/'+DATASETS+'/FREE/'+MODEL_NAME
            if not os.path.exists(save_path_final_free):
                os.makedirs(save_path_final_free)

            # generate meaning maps
            for folder in os.listdir(dataset_path):
                if '.' in folder:
                    continue
                # compute random mean feature
                mean_feature = compute_mean_feature(net,preprocess,device,target_size,MODEL_NAME)
                with torch.no_grad():
                    for DIV_PATCH in DIV_PATCH_LIST:
                        P_N_H=(DIV_PATCH+2)*DIV_STRIDE-DIV_STRIDE-1

                        save_path_free = './outputs/'+DATASETS+'/FREE/'+MODEL_NAME+'_'+str(DIV_PATCH)+'_'+str(DIV_STRIDE)

                        os.makedirs(os.path.join(save_path_free,folder),exist_ok=True)

                        for ith,(img,patches,imgname,ori_size) in enumerate(load_image_pad(os.path.join(dataset_path,folder+'/images/'),
                                                                                patch_div=DIV_PATCH,
                                                                                stride_div=DIV_STRIDE,
                                                                                target_size=target_size,
                                                                                patch_on=True,
                                                                                sample_on=SAMPLE_ON)):

                            # compute clip features
                            image_features = compute_feature(img,net,preprocess,device,MODEL_NAME)

                            # compute patches similarity
                            patches_sims_free=[]
                            for h in range(P_N_H):
                                patches_sims_free.append([])
                                img_tensor =[]
                                for patch in patches[h]:
                                    img_tensor.append(preprocess(patch).to(device))
                                img_tensor = torch.stack(img_tensor)
                                img_tensor = img_tensor.to(device)
                                # compute image features
                                with torch.no_grad():
                                    patch_features_h = net.encode_image(img_tensor)
                                for patch_features in patch_features_h:
                                    sim_free = torch.nn.functional.cosine_similarity(patch_features,image_features-mean_feature).cpu().numpy().flatten()

                                    patches_sims_free[h].append(sim_free)

                            meaning_map_free = compute_meaning_map(patches_sims_free,logit_scale,ori_size)

                            # save as image
                            img_save_free = meaning_map_free / np.max(meaning_map_free) * 255
                            img_save_free = Image.fromarray(img_save_free)
                            img_save_free = img_save_free.convert('L')
                            img_save_free.save(os.path.join(save_path_free,folder,imgname))
                    
                    # add different patch scale
                    if not os.path.exists(os.path.join(save_path_final_free,folder)):
                        os.makedirs(os.path.join(save_path_final_free,folder))
                    for img in os.listdir(os.path.join(dataset_path,folder+'/images/')):
                        maps_free=[]
                        for DIV_PATCH in DIV_PATCH_LIST:
                            save_path_free = './outputs/'+DATASETS+'/FREE/'+MODEL_NAME+'_'+str(DIV_PATCH)+'_'+str(DIV_STRIDE)
                            map_free=np.array(Image.open(os.path.join(save_path_free,folder,img)).convert('L')).astype(np.float32)/255
                            maps_free.append(map_free)
                        maps_free = np.array(maps_free).mean(0)
                        maps_free = maps_free / maps_free.max()
                        maps_free = Image.fromarray((maps_free*255).astype(np.uint8))
                        maps_free.save(os.path.join(save_path_final_free,folder,img))