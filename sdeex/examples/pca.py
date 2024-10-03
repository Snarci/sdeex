import sys
import torch
import torch.nn as nn
from torchvision.transforms import v2
import csv
import os
import argparse

##EXECUTE FROM THE ROOT DIRECTORY
# Add the path to the core functions 
sys.path.append('visualization')
from vit_meta import vit_tiny, vit_small, vit_base
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#core functions
from pca_embedding import get_b_images, embedding_to_image

#arguments for the evaluation
parser = argparse.ArgumentParser(description='PCA Evaluation')
parser.add_argument('--img_path', default='C:/Users/lucat/Documents/RIM/train' , type=str, help='Path to the images')
parser.add_argument('--output', default='F:/sdeex/sdeex/configs/dino_radio/pca_output_rand', type=str, help='Path to the output image')
parser.add_argument('--model', default='dino_vitb16', type=str, help='Model to use')
parser.add_argument('--checkpoint', default='F:/sdeex/sdeex/configs/dino_radio/base_best.pth', type=str, help='Path to the checkpoint')
parser.add_argument('--device', default='cuda', type=str, help='Device to use')

args = parser.parse_args()


def create_model():
    if args.model == 'dino_vitt16':
        model= vit_tiny()
    elif args.model == 'dino_vits16':
        model= vit_small()
    else:
        model= vit_base()
    # load finetuned weights
    pretrained = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    # make correct state dict for loading
    new_state_dict = {}
    pretrained=pretrained['teacher']
    for key, value in pretrained.items():
        if 'dino_head' in key or "ibot_head" in key or"head"in key or  "student" in key or 'loss' in key:
            pass
        else:
            new_key = key.replace('backbone.', '').replace('teacher.', '')
            new_state_dict[new_key] = value
    #model.load_state_dict(new_state_dict, strict=True)
    
    return model.to(args.device).eval()
transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
if __name__ == '__main__':
    img_per_class = 6
    n_classes = 6
    paths = get_b_images(args.img_path, img_per_class, n_classes)
    print('Selected images')
    print('number of images:', len(paths))
    img_size = 224
    model = create_model()
    images_for_plotting = [Image.open(img_path).convert('RGB').resize((img_size, img_size)) for img_path in paths]
    if args.device == 'cuda':
        imgs_tensor = torch.stack([transform(Image.open(img_path).convert('RGB').resize((img_size,img_size))).cuda() for img_path in paths])
    else:    
        imgs_tensor = torch.stack([transform(Image.open(img_path).convert('RGB').resize((img_size,img_size))) for img_path in paths])
    if args.model == 'dino_vitt16':
        size_embedding = 192
    elif args.model == 'dino_vits16':
        size_embedding = 384
    else:
        size_embedding = 768
    with torch.no_grad():
        features = model.forward_features(imgs_tensor)
        pca_features_rgb = embedding_to_image(features, img_per_class*n_classes, img_size, 16, size_embedding,img_per_class)
    #resize the features to the original size 224x224


    print('Saving images')
    #save all the images in the output folder
    #fist clear the folder
    if os.path.exists(args.output):
        for file in os.listdir(args.output):
            os.remove(os.path.join(args.output, file))
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    #plot the images
    fig, ax = plt.subplots(n_classes, img_per_class, figsize=(img_per_class, n_classes))
    if n_classes == 1:
        ax = ax[None, :]

    for i in range(n_classes):
        for j in range(img_per_class):
            ax[i, j].imshow(pca_features_rgb[i * img_per_class + j])
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'pca.png'))
    #save the features using NEAREST interpolation

    for i, img in enumerate(pca_features_rgb):
        img = Image.fromarray((img * 255).astype(np.uint8)).resize((img_size, img_size), Image.NEAREST)
        img.save(os.path.join(args.output, f'pca_{i}.png'))

    for i, img in enumerate(images_for_plotting):
        img.save(os.path.join(args.output, f'img_{i}.png'))
    print('Done')