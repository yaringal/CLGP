import sys; sys.path.append("./")
from CLGP_theano_opt import CLGP_opt
from CLGP_theano_predict import get_error
import numpy as np

def test(N = 100, M = 4, Q = 2, K = 1, D = 3, max_iteration = 20, sx2 = 1):
    # Random initialisation 
    Z = np.random.randn(M,Q)
    m = np.random.randn(N,Q)
    ls = np.zeros((N,Q)) + np.log(0.1)
    mu = np.random.randn(D,M,K) * 1e-2
    lLs = np.zeros((D,M,M))
    lhyp = np.zeros((Q + 2))
    
    params = {'Z': Z, 'm': m, 'ls': ls, 'lL': lLs, 'lhyp': lhyp}
    for i in xrange(D):
        params['mu' + str(i)] = mu[i]

    clgp = CLGP_opt(params, np.array([[0]]))

    # Some synthetic test data from the prior
    obs = []
    for modality in xrange(D):
        S = clgp.estimate(clgp.f['S'], modality=modality)[0]
        obs += [np.array([np.random.multinomial(1, s) for s in S])]
    obs = np.array(obs)
    print np.rollaxis(obs[:,:5,:],1)
    clgp.Y = obs
    clgp.mask = np.zeros((N, D), dtype=bool)
    clgp.mask[:, -1] = False

    print 'Optimising...'
    iteration = 0
    while iteration < max_iteration:
        clgp.opt_one_step(params.keys(), iteration)
        current_ELBO = clgp.ELBO(obs)
        print 'iter ' + str(iteration) + ': ' + str(current_ELBO[0]) + ' +- ' + str(current_ELBO[1])
        if iteration%10 == 0:
            print 'error ' + str(get_error(clgp, clgp.Y, ~clgp.mask)[0])
        iteration += 1

test(N = 100, M = 4, Q = 2, K = 2, D = 3, max_iteration = 100)
