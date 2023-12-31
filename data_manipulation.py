import pandas as pd # for data manipulation

# Visualization
import matplotlib.pyplot as plt # for showing handwritten digits

# Skleran
from sklearn.datasets import load_digits # for MNIST data
from sklearn.model_selection import train_test_split # for splitting data into train and test samples

# UMAP dimensionality reduction
from umap import UMAP

import json

from chart import chart
from embedder import embed_docs

def perform_mapping(file_name, embedding_model):
    with open(file_name, "r") as file:
        data = json.load(file)

    sentences = []
    y=[]
    for d in data:
        sentences.append(d[0])
        y.append(d[1])

    print("EMBEDDING DOCS")
    X = embed_docs(sentences, embedding_model)
    y = pd.Series(y)

    reducer = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                n_components=3, # default 2, The dimension of the space to embed into.
                metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
                n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. 
                learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
                init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
                spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
                set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
                local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
                repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
                negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
                transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
                a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
                angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
                target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
                verbose=False, # default False, Controls verbosity of logging.
                unique=False, # default False, Controls if the rows of your data should be uniqued before being embedded. 
                )

    # Fit and transform the data
    X_trans = reducer.fit_transform(X)

    # Check the shape of the new data
    print('Shape of X_trans: ', X_trans.shape)

    chart(X_trans, y)