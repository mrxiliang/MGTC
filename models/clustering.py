from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import scipy.sparse as sp


def hierarchical_clustering(x,args,  initial_rank=None, distance='cosine', ensure_early_exit=True, verbose=True,  ann_threshold=50000,mode=None,K=None):
    """
    x: input matrix with features in rows.(n_samples, n_features)
    initial_rank: Nx1 first integer neighbor indices (optional). (n_samples, 1)
    req_clust: set output number of clusters (optional).
    distance: one of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    verbose: print verbose output.
    ann_threshold: int (default 40000) Data size threshold below which nearest neighbors are approximated with ANNs.
    """
    # print('Performing finch clustering')
    # req_clust = args.req_clust
    results = {'im2cluster': [], 'centroids': [], 'density': []}

    # x = x.astype(np.float32)
    min_sim = None

    # calculate pairwise similarity orig_dis to find the nearest neighbor and obtain the adj matrix
    adj, orig_dist, first_neighbors, _ = clust_rank(
        x,
        initial_rank,
        distance,
        verbose=verbose,
        ann_threshold=ann_threshold,
        mode = mode,
        K = K
    )

    initial_rank = None


    # obtain clusters by connecting nodes using the adj matrix obtained by cluster_rank
    u, num_clust = get_clust(adj, [], min_sim)

    # group: the parent classes of all subclass nodes, cluster labels, num_cluster: components
    c, mat = get_merge([], u, x) # obtain the centroids according to the partition and raw data

    """find the points farthest from the centroids in each cluster and mask these points in next round of clustering"""


    cluster = defaultdict(list)
    outliers_dist = defaultdict(list)

    for i in range(0, len(u)):  # u: current partition, c: all partitions
        cluster[u[i]].append(i)
        outliers_dist[u[i]].append(i)



    lowest_level_centroids = mat

    ''' save centroids of the bottom layer (layer 0)'''
    lowest_centroids = torch.Tensor(lowest_level_centroids).cuda()
    results['centroids'].append(lowest_centroids)


    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())


    exit_clust = 2
    c_ = c  # transfer value first and then mask

    k = 1
    num_clust = [num_clust] #int->list
    partition_clustering = []
    while exit_clust > 1:
        adj, orig_dist, first_neighbors, knn_index = clust_rank(
            mat,
            initial_rank,
            distance,
            verbose=verbose,
            ann_threshold=ann_threshold
        )

        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)  #u = group

        partition_clustering.append(u)  # all partitions (u: current partition)

        c_, mat = get_merge(c_, u, x)
        c = np.column_stack((c, c_))

        num_clust.append(num_clust_curr)
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust <= 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break


        ''' save the controids of the bottom args.layers '''


        ''' save the controids at args.layers '''
        # if args.layers=3 means: save 533 131 32  from [533, 131, 32, 7, 2]
        if k < args.layers:
            centroids = torch.Tensor(mat).cuda()
            results['centroids'].append(centroids)

        k += 1



    """ save multiple partitions """
    # save 131 32 7 from [533, 131, 32, 7, 2]
    for i in range(0, args.layers):
        im2cluster = [int(n[i]) for n in c]
        im2cluster = torch.LongTensor(im2cluster).cuda()
        results['im2cluster'].append(im2cluster)

    return c, num_clust, partition_clustering, lowest_level_centroids, results


def clust_rank(
        mat,
        initial_rank=None,
        metric='cosine',
        verbose=False,
        ann_threshold=50000,
        mode=None,
        K=None):
    knn_index = None
    s = mat.shape[0]
    #mat = mat.reshape(s,-1)
    if initial_rank is not None:
        orig_dist = []
    elif s <= ann_threshold:
        # If the sample size is smaller than threshold, use metric to calculate similarity.
        # If the sample size is larger than threshold, use PyNNDecent to speed up the calculation of nearest neighbor

        if not np.isfinite(mat).all():
            mat[np.isnan(mat)] = mat.mean()
            mat[np.isinf(mat)] = mat.mean()

        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=metric)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        # if verbose:
        #     print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        mat = np.array(mat, dtype=np.float32)
        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=metric,
            verbose=verbose)
        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12


    sparce_adjacency_matrix = sp.csr_matrix(
        (np.ones_like(initial_rank, dtype=np.float32),
        (np.arange(0, s), initial_rank)),
         shape=(s, s))  # join adjacency matrix based on Initial rank


    return sparce_adjacency_matrix, orig_dist, initial_rank, knn_index

def get_clust(a, orig_dist, min_sim=None):
    # connect nodes based on adj, orig_dist, min_sim
    # build the graph and obtain multiple components/clusters
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)

    return u, num_clust


def get_merge(partition, group, data):
    # get_merge([], group, x)
    # u/group: (n,)  data/x: (n, dim)
    if len(partition) != 0:
        _, ig = np.unique(partition, return_inverse=True)
        partition = group[ig]
    else:
        partition = group

    mat = cool_mean(data, partition, max_dis_list=None) # mat: computed centroids(k,dim)
    # data: (n, dim)   partition: (n,)  return:(k, dim)
    return partition, mat

def cool_mean(data, partition, max_dis_list=None):
    s = data.shape[0]
    un, nf = np.unique(partition, return_counts=True)

    row = np.arange(0, s)
    col = partition
    d = np.ones(s, dtype='float32')

    if max_dis_list is not None:
        for i in max_dis_list:
            data[i] = 0
        nf = nf - 1

    umat = sp.csr_matrix((d, (row, col)), shape=(s, len(un)))
    cluster_rep = umat.T @ data.reshape(s,-1)
    a = nf[..., np.newaxis]
    cluster_mean_rep = cluster_rep / nf[..., np.newaxis]

    return cluster_mean_rep

