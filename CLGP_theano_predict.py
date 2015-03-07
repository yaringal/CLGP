import numpy as np
from CLGP_theano_opt import *

def get_error(clgp, Y_true, mask, start = 0, samples = 100, shuffle = False, shuffle_min = 0.05):
    orig_params = {'m': clgp.params['m'], 'ls': clgp.params['ls']}
    N, D = Y_true[0].shape[0], len(Y_true)
    clgp.params['m'] = clgp.params['m'][start:start+N]
    clgp.params['ls'] = clgp.params['ls'][start:start+N]

    probs = np.ones((N))
    acc_err, nMiss = 0, 0
    for modality in xrange(D):
        m = mask[:, modality]
        if m.sum() == 0: continue
        S, _ = clgp.estimate(clgp.f['S'], modality=modality, samples=samples)
        prob = np.sum(Y_true[modality] * S, 1)[m]
        #print prob[:20] #print S
        acc_err += np.sum(np.log2(prob))
        nMiss += m.sum()
        probs[m] = np.minimum(probs[m], prob)

    clgp.params['m'], clgp.params['ls'] = orig_params['m'], orig_params['ls']
    if shuffle:
        shuffle_mask = (probs < shuffle_min).nonzero()[0] + start
        return 2**(-acc_err/nMiss), shuffle_mask
    return 2**(-acc_err/nMiss), []

def get_error_with_std(clgp, Y_true, mask, start = 0, samples = 100, shuffle = False, shuffle_min = 0.05):
    orig_params = {'m': clgp.params['m'], 'ls': clgp.params['ls']}
    N, D = Y_true[0].shape[0], len(Y_true)
    clgp.params['m'] = clgp.params['m'][start:start+N]
    clgp.params['ls'] = clgp.params['ls'][start:start+N]

    probs = np.ones((N))
    acc_err, acc_err2, nMiss = 0, 0, 0
    for modality in xrange(D):
        m = mask[:, modality]
        if m.sum() == 0: continue
        S, _ = clgp.estimate(clgp.f['S'], modality=modality, samples=samples)
        prob = np.sum(Y_true[modality] * S, 1)[m]
        print prob[:20] #print S
        acc_err += np.sum(np.log2(prob))
        acc_err2 += np.sum(np.log2(prob)**2)
        nMiss += m.sum()
        probs[m] = np.minimum(probs[m], prob)

    clgp.params['m'], clgp.params['ls'] = orig_params['m'], orig_params['ls']
    if shuffle:
        shuffle_mask = (probs < shuffle_min).nonzero()[0] + start
        return -acc_err/nMiss, np.sqrt((acc_err2/nMiss - (acc_err/nMiss)**2) / (nMiss - 1)), shuffle_mask
    return -acc_err/nMiss, np.sqrt((acc_err2/nMiss - (acc_err/nMiss)**2) / (nMiss - 1)), []
