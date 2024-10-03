import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision.transforms import v2

##EXECUTE FROM THE ROOT DIRECTORY
# Add the path to the core functions 
sys.path.append('core')
sys.path.append('evaluation')
from vit_meta import vit_tiny
#core functions
from model_wrapper import ModelWrapper
from image_dataset import get_image_dataset
#evaluation functions
from extract_features import extract_features

#arguments
parser = argparse.ArgumentParser(description='DINO Radio')
parser.add_argument('--model', default='dino_vits16', type=str, help='Model name')
parser.add_argument('--checkpoint', default=None, type=str, help='Path to the checkpoint')
parser.add_argument('--dataset_path_train', type=str, help='Path to the dataset for training')
parser.add_argument('--dataset_path_test', type=str, help='Path to the dataset for testing')
parser.add_argument('--output_path', type=str, help='Path to the output folder')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
parser.add_argument('--device', default='cuda', type=str, help='Device')

args = parser.parse_args()


#model creation
def create_model():
    if args.model == 'dino_vitt16':
        model= vit_tiny()
    else:
        model= torch.hub.load('facebookresearch/dino:main', args.model)
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
    model.load_state_dict(new_state_dict, strict=True)
    # wrap the model
    wrapperd_model = ModelWrapper(model)
    wrapperd_model = wrapperd_model.to(args.device).eval()
    return wrapperd_model

#dataset creation
def create_dataset(path):
    # default transform for dino models
    DEFAULT_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_STD = (0.229, 0.224, 0.225)
    transform = v2.Compose([
        v2.ToImage(), 
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)])
    #args for dataset
    dataset_args = argparse.Namespace(
        data_dir=path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform,
        persistent_workers= False,
        shuffle=False)
    #get dataset
    dataset = get_image_dataset(dataset_args)
    return dataset



if __name__ == '__main__':
    #create model
    model = create_model()
    #create dataset
    train_loader = create_dataset(args.dataset_path_train)
    test_loader = create_dataset(args.dataset_path_test)
    #get classes
    classes = train_loader.dataset.classes
    #extract features
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    #extract features
    model_name = args.model
    extract_features(model, test_loader, classes, f"{args.output_path}/radio_{model_name}_test.h5")
    extract_features(model, train_loader, classes, f"{args.output_path}/radio_{model_name}_train.h5")