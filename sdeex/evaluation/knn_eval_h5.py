import h5py as h5
from sklearn.neighbors import KNeighborsClassifier
#get metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import os
import argparse

#arguments for the evaluation
parser = argparse.ArgumentParser(description='KNN Evaluation')
parser.add_argument('--model', type=str, help='Model name')
parser.add_argument('--features_folder', type=str, help='Path to the features folder')
parser.add_argument('--features', type=str, help='Path to the features without train or test')
parser.add_argument('--output', type=str, help='Path to the output csv')
parser.add_argument('--k', type=int, help='Number of neighbors')
args = parser.parse_args()

def load_h5_file(file_path):
    with h5.File(file_path, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
    return X, y

def knn_eval_h5(args):
    #load the features
    X_train, y_train = load_h5_file(os.path.join(args.features_folder, args.features + "_train.h5"))
    X_test, y_test = load_h5_file(os.path.join(args.features_folder, args.features + "_test.h5"))
    #create the model
    model = KNeighborsClassifier(n_neighbors=args.k)
    #fit the model
    model.fit(X_train, y_train)
    #predict the test data
    print("Predicting")
    y_pred = model.predict(X_test)
    #get the metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    #save the metrics
    #save results in a csv if it does not exist also create the header
    if not os.path.exists(args.output):
        with open("results_new.csv", "w") as f:
            f.write("Model,K,Accuracy,Precision,Recall,F1\n")

    #save the results
    with open(args.output, "a") as f:
        f.write(f"{args.model},{args.k},{accuracy},{precision},{recall},{f1}\n")
        print("results saved")