import argparse
import umap.umap_ as umap
import matplotlib.pyplot as plt
import h5py as h5
import os 
 


def load_h5_file(file_path):
    with h5.File(file_path, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
    return X, y

def umap_eval_h5(test_features, output):
    #load the features
    
    X_test, y_test = load_h5_file(test_features)
    #fit the umap model
    embedding = umap.UMAP(a=None, angular_rp_forest=False, b=None,
    force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
    local_connectivity=1.0, low_memory=False, metric='euclidean',
    metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
    n_neighbors=120, negative_sample_rate=5, output_metric='euclidean',
    output_metric_kwds=None, repulsion_strength=1.0,
    set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
    target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
    transform_queue_size=4.0, unique=False, verbose=False,n_jobs=-1).fit_transform(X_test)
    #classes are the unique numbers in y
    classes = list(set(y_test))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_test, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar()
    filename = 'umap_3.png'
    full_path = os.path.join(output, filename)
    if not os.path.exists(output):
        os.makedirs(output)
    plt.savefig(full_path)


if __name__ == '__main__':
    #arguments for the evaluation
    parser = argparse.ArgumentParser(description='UMAP Evaluation')
    parser.add_argument('--test_features', type=str, help='Path to the test features')
    parser.add_argument('--output', type=str, help='Path to the output image')
    args, unknown = parser.parse_known_args()
    umap_eval_h5(args.test_features, args.output)

    



