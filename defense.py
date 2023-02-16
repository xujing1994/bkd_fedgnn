import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from scipy.special import logit
import sklearn.metrics.pairwise as smp

def fedavg(args, grad_in):
    grad = np.array(grad_in).reshape((args.num_workers, -1)).mean(axis=0)
    return grad.tolist()

def foolsgold(args, grad_history, grad_in):
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
