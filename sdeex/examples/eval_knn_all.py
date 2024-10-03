import sys
import argparse 
sys.path.append('evaluation')

from knn_eval_h5 import knn_eval_h5


MODEL_NAMES = ['radio_dino_vitt16', 'radio_dino_vits16','radio_dino_vitb16']
KS = [1, 5, 20]
FEATURE_PATH = 'E:/DinoRAD/evaluation/extracted_features'   
CSV_PATH = 'E:/DinoRAD/evaluation/results_new.csv'

for k in KS:
    for model in MODEL_NAMES:
        args = argparse.Namespace(
            model=model,
            features_folder=FEATURE_PATH,
            features=model,
            output=CSV_PATH,
            k=k
        )
        knn_eval_h5(args)
        print(f"Model: {model}, K: {k} done!")