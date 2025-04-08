# Compute free-viewing meaningmap with different models
# patch and stride size get from dataset_info.json
from util.image_loader import load_image_pad, other_maps_scale
import open_clip
import torch
import numpy as np
from PIL import Image
import os
from util import utils
import cv2
import datetime
from transformers import ViTModel, ViTImageProcessor
import json
import timm

MODEL_NAMES=['laion/CLIP-ViT-B-16-laion2B-s34B-b88K']
# MODEL_NAMES=['laion/CLIP-ViT-L-14-laion2B-s32B-b82K','laion/CLIP-ViT-H-14-laion2B-s32B-b79K','laion/CLIP-ViT-bigG-14-laion2B-39B-b160k']
# MODEL_NAMES=['laion/CLIP-ViT-B-16-laion2B-s34B-b88K','laion/CLIP-ViT-L-14-laion2B-s32B-b82K','laion/CLIP-ViT-H-14-laion2B-s32B-b79K']
# MODEL_NAMES=['laion/CLIP-ViT-B-16-laion400m_e31']
# MODEL_NAMES=['laion/CLIP-ViT-bigG-14-laion2B-39B-b160k']

# NAME='SUBWAY'
# NAME = 'DREYEVE'
NAMES = ['SUBWAY']
# NAMES = ['DREYEVE_roof_camera']
# NAMES = ['THUE_TASK']
# NAME='Hollywood2'
# NAME = 'DHF1K'
# NAME = 'UCF'
# NAME = 'THUE_FREE'

SAMPLE_ON=True

# logit scale does not affect the AUC and sAUC results, just for visualization
# LOGIT_SCALE=10
# TASK='FREE'
MODE='cosine'

# define a feature computation function
def compute_feature(img,net,preprocess,device,model_name):
    if model_name == 'facebook/dinov2-large' or model_name == 'google/vit-large-patch16-224-in21k' \
                or model_name == 'OpenGVLab/InternViT-6B-448px-V1-2':
        img = preprocess(img, return_tensors="pt").pixel_values.to(device)
        if model_name == 'OpenGVLab/InternViT-6B-448px-V1-2':
            img = img.to(torch.bfloat16)
        with torch.no_grad():
            image_features = net(img).pooler_output
            if model_name == 'OpenGVLab/InternViT-6B-448px-V1-2':
                image_features = image_features.to(torch.float16)
    
    elif model_name == 'vit_base_patch16_224.mae':
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = net(img)

    elif model_name == 'nomic-ai/nomic-embed-vision-v1.5':
        img_clip = preprocess(img,return_tensors="pt")
        img_clip = img_clip.to(device)
        with torch.no_grad():
            image_features = net(**img_clip).last_hidden_state
            # get cls token
            image_features = image_features[:,0,:].squeeze(1)

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

    # center_size=(TARGET_SIZE[0]*(DIV_PATCH-1),TARGET_SIZE[1]*(DIV_PATCH-1))

    # meaning_map = cv2.resize(patches_sims,center_size,interpolation=cv2.INTER_CUBIC)
    # # padding to original size, where pad value is nearest value
    # meaning_map = np.pad(meaning_map,((TARGET_SIZE[0]//2,TARGET_SIZE[0]//2),(TARGET_SIZE[0]//2,TARGET_SIZE[0]//2)),mode='edge')

    meaning_map=patches_sims
    # softmax
    meaning_map = np.exp(meaning_map * logit_scale)
    meaning_map = cv2.resize(meaning_map,ori_size,interpolation=cv2.INTER_CUBIC)
    # meaning_map = cv2.GaussianBlur(meaning_map,(15,15), 10,10)
    meaning_map = meaning_map / np.max(meaning_map)
    # meaning_map = np.array(meaning_map).astype(np.float32)/255
    return meaning_map

def load_model(model_name):
    if model_name == 'google/vit-base-patch16-224-in21k':
        net = ViTModel.from_pretrained(model_name)
        preprocess = ViTImageProcessor.from_pretrained(model_name)
    elif model_name == 'vit_base_patch16_224.mae':
        net = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
        )
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(net)
        preprocess = timm.data.create_transform(**data_config, is_training=False)
    elif model_name == 'laion/CLIP-ViT-B-16-laion400m_e31':
        net,_, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e31')
    elif 'laion' in model_name:
        net, preprocess = open_clip.create_model_from_pretrained('hf-hub:'+model_name)
    
    return net, preprocess

def compute_mean_feature(net,preprocess,device,target_size=None,model_name=None):
    with torch.no_grad():
        feat_num=1000
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

        mean_txt_feature = None
        for i in range(feat_num):
            txt = np.random.randint(0,1000,(1,1),dtype=np.int64)
            txt = torch.from_numpy(txt)
            txt = txt.to(device)
            txt_feature = net.encode_text(txt)
            if mean_txt_feature is None:
                mean_txt_feature = txt_feature
            else:
                mean_txt_feature += txt_feature
        mean_txt_feature /= feat_num

        return mean_feature, mean_txt_feature

def load_tokenizer(name):
    # if name == 'laion/CLIP-ViT-B-16-laion2B-s34B-b88K':
    #     tokenizer = open_clip.get_tokenizer('ViT-B-16')
    # if name == 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k':
    #     tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    # if name == 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K':
    #     tokenizer = open_clip.get_tokenizer('ViT-L-14')
    # if name == 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K':
    #     tokenizer = open_clip.get_tokenizer('ViT-H-14')
    tokenizer = open_clip.get_tokenizer('hf-hub:'+name)
    return tokenizer

if __name__ == '__main__':
    for NAME in NAMES:
        for MODEL_NAME in MODEL_NAMES:
            # load dataset_info
            with open('./results/dataset_info.json','r') as f:
                datasets_info = json.load(f)

            # get parameters
            DIV_PATCH_LIST=datasets_info[NAME]['DIV_PATCH_LIST']
            DIV_STRIDE=datasets_info[NAME]['DIV_STRIDE']
            dataset_path=datasets_info[NAME]['DATASET_PATH']

            # load task json
            # get the parent folder of dataset_path
            parent_folder = os.path.dirname(dataset_path)
            parent_folder = os.path.dirname(parent_folder)
            with open(parent_folder+'/task.json','r') as f:
                task_info = json.load(f)
            
            target_size = (224,224)
            
            # load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            net, preprocess = load_model(MODEL_NAME)
            # tokenizer = open_clip.get_tokenizer('ViT-B-16')
            tokenizer = load_tokenizer(MODEL_NAME)
            net.to(device).eval()

            # get logit_scale
            logit_scale = net.logit_scale.item()

            for DIV_PATCH in DIV_PATCH_LIST:
                P_N_H=(DIV_PATCH+2)*DIV_STRIDE-DIV_STRIDE-1
                P_N_W=(DIV_PATCH+2)*DIV_STRIDE-DIV_STRIDE-1
                SAVE_NAME=datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                save_path_task = './outputs/'+NAME+'/TASK/'+MODEL_NAME+'_'+MODE+'_'+str(DIV_PATCH)+'_'+str(DIV_STRIDE)
                save_path_free = './outputs/'+NAME+'/FREE/'+MODEL_NAME+'_'+MODE+'_'+str(DIV_PATCH)+'_'+str(DIV_STRIDE)

                print("model_name: ",MODEL_NAME)
                print("save_path: ",save_path_free)

                # copy current code to save_path
                if not os.path.exists(save_path_free):
                    os.makedirs(save_path_free)
                if not os.path.exists(save_path_task):
                    os.makedirs(save_path_task)
                with open(os.path.join(save_path_free,SAVE_NAME+'.py'),'w') as f:
                    with open(__file__,'r') as f2:
                        f.write(f2.read())

                # compute random mean feature
                mean_feature, _ = compute_mean_feature(net,preprocess,device,target_size,MODEL_NAME)

                with torch.no_grad():
                    for folder in os.listdir(dataset_path):
                        if '.' in folder:
                            continue
                        if 'roof_camera' in folder and (int(folder)<40 or int(folder)>60):
                            continue
                        tok = tokenizer([task_info[folder]]).to(device)
                        task_feature = net.encode_text(tok).detach()

                        os.makedirs(os.path.join(save_path_free,folder),exist_ok=True)
                        os.makedirs(os.path.join(save_path_task,folder),exist_ok=True)
                        # if SAMPLE_ON:
                        #     sample_count=0
                        for ith,(img,patches,imgname,ori_size) in enumerate(load_image_pad(os.path.join(dataset_path,folder+'/images/'),
                                                                                patch_div=DIV_PATCH,
                                                                                stride_div=DIV_STRIDE,
                                                                                target_size=target_size,
                                                                                patch_on=True,
                                                                                sample_on=SAMPLE_ON)):
                            
                            # if os.path.exists(os.path.join(save_path_free,folder,imgname)):
                            #     continue
                            
                            # if SAMPLE_ON:
                            #     sample_count+=1
                            #     if sample_count %10 !=0:
                            #         continue

                            # compute clip features
                            image_features = compute_feature(img,net,preprocess,device,MODEL_NAME)

                            # compute patches similarity
                            patches_sims_task=[]
                            patches_sims_free=[]
                            for h in range(P_N_H):
                                patches_sims_free.append([])
                                patches_sims_task.append([])
                                img_tensor =[]
                                for patch in patches[h]:
                                    img_tensor.append(preprocess(patch).to(device))
                                img_tensor = torch.stack(img_tensor)
                                img_tensor = img_tensor.to(device)
                                # compute image features
                                with torch.no_grad():
                                    patch_features_h = net.encode_image(img_tensor)
                                for patch_features in patch_features_h:
                                    if MODE=='cosine':
                                        sim_task = torch.nn.functional.cosine_similarity(patch_features,task_feature-mean_feature).cpu().numpy().flatten()
                                        sim_free = torch.nn.functional.cosine_similarity(patch_features,image_features-mean_feature).cpu().numpy().flatten()

                                    else:
                                        raise NotImplementedError

                                    patches_sims_free[h].append(sim_free)
                                    patches_sims_task[h].append(sim_task)

                            meaning_map_free = compute_meaning_map(patches_sims_free,logit_scale,ori_size)
                            meaning_map_task = compute_meaning_map(patches_sims_task,logit_scale,ori_size)

                            # save as image
                            img_save_task = meaning_map_task / np.max(meaning_map_task) * 255
                            img_save_task = Image.fromarray(img_save_task)
                            img_save_task = img_save_task.convert('L')
                            img_save_task.save(os.path.join(save_path_task,folder,imgname))

                            img_save_free = meaning_map_free / np.max(meaning_map_free) * 255
                            img_save_free = Image.fromarray(img_save_free)
                            img_save_free = img_save_free.convert('L')
                            img_save_free.save(os.path.join(save_path_free,folder,imgname))