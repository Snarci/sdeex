# PCA for feature inferred
from sklearn.decomposition import PCA
import os
import random
from glob import glob
import numpy as np

def embedding_to_image(features, b_size, img_size,patch_size, emb_size,class_num=1):
    patch_num_h = img_size//patch_size
    patch_num_w = img_size//patch_size
    num_patches = patch_num_h*patch_num_w
    images_class = b_size//class_num
    print(features.shape)
    
    pca_features= []
    for i in range(class_num):
        pca = PCA(n_components=3)
        current_class = features[i*images_class:(i+1)*images_class]
        current_class = current_class.reshape(images_class * num_patches, emb_size).cpu().numpy()
        pca.fit(current_class)
        pca_features.append(pca.transform(current_class))


    pca_features = np.concatenate(pca_features, axis=0)


    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())

    pca_features_rgb = pca_features.copy()
    pca_features_rgb = pca_features_rgb.reshape(b_size, patch_num_h, patch_num_w, 3)

    return pca_features_rgb


def get_b_images( img_path, n_images_class,n_classes_selected,extension='png'):
    classes = os.listdir(img_path)
    #set seed as current time
    random.seed(42)
    #shuffle classes
    paths = []
    random.shuffle(classes)
    classes = classes[:n_classes_selected]
    if len(os.listdir(img_path)) < n_classes_selected:
        print('Number of classes is less than the number of classes selected')
        return paths
    if n_images_class > len(os.listdir(os.path.join(img_path, classes[0]))):
        print('Number of images in the class is less than the number of images selected')
        return paths
    for class_name in classes:
        class_path = os.path.join(img_path, class_name)
        images = glob(os.path.join(class_path, f'*.{extension}'))
        random.shuffle(images)
        paths += images[:n_images_class]
    return paths

