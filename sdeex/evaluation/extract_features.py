import torch
import numpy as np
import h5py as h5
from tqdm import tqdm

def extract_features(model, loader,classes,name,check=True):
    X = []
    y = []
    with torch.no_grad():
        for img, label in tqdm(loader):
            img = img.cuda()
            features = model(img)
            #if the shape is 3d avg the second dimension and get a 2d tensor 
            #otherwise it would be messy to store such high dimensional data
            if len(features.shape) == 3:
                features = features.mean(1)
            if check:
                print("Feature shape:", features.shape)
                print("Feature type:", type(features))
            for j in range(features.shape[0]):
                X.append(features[j].cpu().numpy())
                y.append(classes[label[j].item()])
            if check:
                print("Feature shape:", features.shape)
                print("X shape:", np.array(X).shape)
                check=False
    X = np.array(X)
    y = np.array(y)
    y_int = np.zeros(y.shape)
    for i, c in enumerate(classes):
        y_int[y == c] = i
    #save the features
    with h5.File(name, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y_int)
    print("Saved features to", name)
    return X, y

