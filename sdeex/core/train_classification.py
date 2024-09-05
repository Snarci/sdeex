import sys
import argparse
import torch
import torch.nn as nn
from torchvision.transforms import v2
import csv
import os
##EXECUTE FROM THE ROOT DIRECTORY
# Add the path to the core functions 
import numpy as np
#core functions
from model_wrapper import ModelWrapper
from image_dataset import get_image_dataset
from classification_utils import train_loop, eval_loop,get_metrics
from torch.utils.data import default_collate

#arguments
parser = argparse.ArgumentParser(description='DINO Radio')
parser.add_argument('--model', default='dino_vits16', type=str, help='Model name')
parser.add_argument('--checkpoint', default=None, type=str, help='Path to the checkpoint')
parser.add_argument('--dataset_path_train', type=str, help='Path to the dataset for training')
parser.add_argument('--dataset_path_val', type=str, help='Path to the dataset for validation')
parser.add_argument('--dataset_path_test', type=str, help='Path to the dataset for testing')
parser.add_argument('--output_path', type=str, help='Path to the output folder')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
parser.add_argument('--device', default='cuda', type=str, help='Device')
parser.add_argument('--epochs', default=30, type=int, help='Number of epochs')
parser.add_argument('--warmup_epochs', default=5, type=int, help='Number of warmup epochs')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--dataset_name', type=str, help='Dataset')
parser.add_argument('--use_amp', default=True, type=bool, help='Use automatic mixed precision')
parser.add_argument('--gradient_clip', default=1, type=float, help='Gradient clipping')
parser.add_argument('--patience', default=30, type=int, help='Patience for early stopping')
parser.add_argument('--cutmix', default=False, type=bool, help='Use cutmix')
parser.add_argument('--mixup', default=False, type=bool, help='Use mixup')
parser.add_argument('--seed', default=-1, type=int, help='Seed')
parser.add_argument('--log_file', default='log.csv', type=str, help='Log file')
parser.add_argument('--result_file', default='results.csv', type=str, help='Result file')

args = parser.parse_args()

#set seed
if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

#model creation
def create_model():
    '''
    Create the model
    adapt this function to create the model you want to use
    Returns: model
    '''
    return model
    
def cutmix_mixup(classes, prob=0.5):
        cutmix = v2.CutMix(num_classes=len(classes), alpha=0.99)
        mixup = v2.MixUp(num_classes=len(classes), alpha=0.99)
        identity = v2.Identity()
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup, identity], p=[0.25, 0.25, 0.5])
        if args.cutmix and args.mixup:
            print("Using both CutMix and MixUp")
            def collate_fn(batch):
                return  cutmix_or_mixup(*default_collate(batch))
        elif args.cutmix:
            print("Using CutMix")
            def collate_fn(batch):
                return cutmix(*default_collate(batch))
        elif args.mixup:
            print("Using MixUp")
            def collate_fn(batch):
                return mixup(*default_collate(batch))
        else:
            print("Using default collate function")
            def collate_fn(batch):
                return default_collate(batch)
        return collate_fn
#dataset creation
def create_dataset(path, is_train):
    # default transform for dino models
    DEFAULT_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_STD = (0.229, 0.224, 0.225)
    if is_train:
        transform = v2.Compose([
            v2.ToImage(), 
            v2.Resize((224, 224)),
            v2.AutoAugment(),
            #v2.RandomHorizontalFlip(),
            #v2.RandomVerticalFlip(),
            #v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # Random crop and resize to the original size
            #v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add random changes in brightness, contrast, saturation, and hue
            #v2.RandomRotation(degrees=10),  # Randomly rotate the image
            #v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Random affine transformation
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
        ])
    else:
        transform = v2.Compose([
            v2.ToImage(), 
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
    ])
    #args for dataset
    dataset_args = argparse.Namespace(
        data_dir=path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform,
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=True if is_train else False)
    #get dataset
    dataset = get_image_dataset(dataset_args)
    return dataset
if __name__ == '__main__':
    #print args info in a good format
    print("Arguments:")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    #create output path
    #create dataset
    train_loader = create_dataset(args.dataset_path_train, is_train=True)
    val_loader = create_dataset(args.dataset_path_val, is_train=False)
    test_loader = create_dataset(args.dataset_path_test, is_train=False)
    #get classes
    classes = train_loader.dataset.classes
    #cutmix and mixup
    collate_fn = cutmix_mixup(classes)
    if args.cutmix or args.mixup:
        train_loader.collate_fn = collate_fn

    print("Classes:", classes)
    log_file = args.log_file
    result_file = args.result_file
    args.output_path = os.path.join(args.output_path, args.dataset_name)
    log_file = os.path.join(args.output_path, args.model, log_file)
    #create model
    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup phase with a smaller learning rate
    warmup_lr = args.lr * 0.05  # Define a lower learning rate for warmup, e.g., 10% of the original LR
    warmup_optimizer = torch.optim.AdamW(model.parameters(), lr=warmup_lr)
    best_val_auc = 0
    for epoch in range(args.warmup_epochs):
        train_loss = train_loop(model, train_loader, warmup_optimizer, criterion, args.device, use_amp=args.use_amp, gradient_clip=args.gradient_clip)
        val_loss, logits, targets = eval_loop(model, val_loader, criterion, args.device)
        accuracy, precision, recall, f1, auc = get_metrics(logits, targets)
        print(f'Epoch {epoch+1}/{args.warmup_epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1} | AUC: {auc} | LR: {scheduler.get_last_lr()}')

    # Normal training phase with the original learning rate and scheduler
    patience = args.patience
    last_patience_losses = []
    for epoch in range(args.warmup_epochs, args.epochs):
        train_loss = train_loop(model, train_loader, optimizer, criterion, args.device, use_amp=args.use_amp, gradient_clip=args.gradient_clip)
        val_loss, logits, targets = eval_loop(model, val_loader, criterion, args.device)
        accuracy, precision, recall, f1, auc = get_metrics(logits, targets)
        print(f'Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1} | AUC: {auc} | LR: {scheduler.get_last_lr()}')
        # Save the model if the AUC is higher than the previous best
        if auc >= best_val_auc:
            best_val_auc = auc
            if not os.path.exists(f'{args.output_path}/{args.model}'):
                os.makedirs(f'{args.output_path}/{args.model}')
            torch.save(model.state_dict(), f'{args.output_path}/{args.model}/best.pth')
        #log the results to a csv
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, args.model, accuracy, precision, recall, f1, auc])
        # Early stopping
        if len(last_patience_losses) == patience:
            if val_loss > min(last_patience_losses):
                print(f'Early stopping at epoch {epoch+1}')
                break
            last_patience_losses.pop(0)  # Remove the first element of the list


        last_patience_losses.append(val_loss)  # Add the current validation loss to the list

        # Step the scheduler
        scheduler.step()
    # test on the test set
    model.load_state_dict(torch.load(f'{args.output_path}/{args.model}/best.pth'))
    test_loss, logits, targets = eval_loop(model, test_loader, criterion, args.device)
    accuracy, precision, recall, f1, auc = get_metrics(logits, targets)
    print(f'Test Loss: {test_loss} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1} | AUC: {auc}')
    # Log the results
    if not os.path.exists(result_file):
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset','Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.dataset_name, args.model, accuracy, precision, recall, f1, auc])