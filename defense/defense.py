import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from scipy.special import logit
import sklearn.metrics.pairwise as smp

def flame(args, grad_in):
    grad_in = np.array(grad_in).reshape((args.num_workers, -1))
    distance_maxtrix = pairwise_distances(grad_in, metric='cosine')
    cluster = hdbscan.HDBSCAN(metric='l2', min_cluster_size=3, allow_single_cluster=True)
    cluster.fit(distance_maxtrix)
    label = cluster.labels_
    print(label)
    if (label == -1).all():
        bengin_id = np.arange(args.num_workers).tolist()
    else:
        label_class, label_count = np.unique(label, return_counts=True)
        if -1 in label_class:
            label_class, label_count = label_class[1:], label_count[1:]
        majority = label_class[np.argmax(label_count)]
        bengin_id = np.where(label == majority)[0].tolist()
        print(bengin_id)
    grad = (grad_in[bengin_id].sum(axis=0)) / len(bengin_id)
    return grad.tolist()

def fedavg(args, grad_in):
    grad = np.array(grad_in).reshape((args.num_workers, -1)).mean(axis=0)
    return grad.tolist()

def foolsgold(args, grad_history, grad_in):
    grad_in = np.array(grad_in).reshape((args.num_workers, -1))
    grad_history = np.array(grad_history)
    if grad_history.shape[0] != args.num_workers:
        grad_history = grad_history[:args.num_workers,:] + grad_history[args.num_workers:,:]

    distance_maxtrix = pairwise_distances(grad_history, metric='cosine')
    similarity_maxtrix = 1 -distance_maxtrix

    mv = np.sort(similarity_maxtrix, axis=0)[-2]


    alpha = np.zeros(mv.shape)
    for i in range(args.num_workers):
        for j in range(args.num_workers):
            if mv[j] > mv[i]:
                similarity_maxtrix[i,j] *= mv[i]/mv[j]
        alpha[i] = 1 - np.sort(similarity_maxtrix[i])[-2]
    alpha = alpha/np.max(alpha)
    alpha = logit(alpha)+0.5
    alpha[np.where(alpha==np.inf)]=1
    grad = np.average(grad_in, weights=alpha, axis=0)
    return grad.tolist(), grad_history.tolist()

def foolsgold_jx(args, grad_history, grad_in):
    epsilon = 1e-5
    grad_in = np.array(grad_in).reshape((args.num_workers, -1))
    grad_history = np.array(grad_history)
    if grad_history.shape[0] != args.num_workers:
        grad_history = grad_history[:args.num_workers,:] + grad_history[args.num_workers:,:]

    similarity_maxtrix = smp.cosine_similarity(grad_history) - np.eye(args.num_workers)

    mv = np.max(similarity_maxtrix, axis=1) + epsilon

    alpha = np.zeros(mv.shape)
    for i in range(args.num_workers):
        for j in range(args.num_workers):
            if mv[j] > mv[i]:
                similarity_maxtrix[i,j] *= mv[i]/mv[j]

    alpha = 1 - (np.max(similarity_maxtrix, axis=1))
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    alpha = alpha/np.max(alpha)
    alpha[(alpha == 1)] = 0.99
    alpha = (np.log((alpha / (1 - alpha)) + epsilon) + 0.5)
    alpha[(np.isinf(alpha) + alpha > 1)] = 1
    alpha[(alpha < 0)] = 0
    print("alpha:")
    print(alpha)
    grad = np.average(grad_in, weights=alpha, axis=0)
    return grad.tolist(), grad_history.tolist(), alpha
