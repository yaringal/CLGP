# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
import numpy as np
import cPickle

print 'Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False

eps = 1e-4
class kernel:
    def RBF(self, sf2, l, X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        dist = ((X1 / l)**2).sum(1)[:, None] + ((_X2 / l)**2).sum(1)[None, :] - 2*(X1 / l).dot((_X2 / l).T)
        RBF = sf2 * T.exp(-dist / 2.0)
        return (RBF + eps * T.eye(X1.shape[0])) if X2 is None else RBF
    def RBFnn(self, sf2, l, X):
        return sf2 + eps
    def LIN(self, sl2, X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        LIN = sl2 * (X1.dot(_X2.T) + 1)
        return (LIN + eps * T.eye(X1.shape[0])) if X2 is None else LIN
    def LINnn(self, sl2, X):
        return sl2 * (T.sum(X**2, 1) + 1) + eps

class CLGP_model:
    def __init__(self, params, sx2 = 1, linear_model = False, samples = 20, use_hat = False):
        ker, self.samples, self.params, self.KmmInv  = kernel(), samples, params, {}
        self.use_hat = use_hat

        model_file_name = 'model' + ('_hat' if use_hat else '') + ('_linear' if linear_model else '') + '.save'

        try:
            print 'Trying to load model...'
            with open(model_file_name, 'rb') as file_handle:
                obj = cPickle.load(file_handle)
                self.f, self.g, self.f_Kmm, self.f_KmmInv, self.dKmm_d = obj
                self.update_KmmInv_cache()
                print 'Loaded!'
            return
        except:
            print 'Failed. Creating a new model...'

        Y, Z, m, ls, mu, lL, eps_MK, eps_NQ, eps_NK, KmmInv = T.dmatrices('Y', 'Z', 'm', 'ls', 'mu', 
            'lL', 'eps_MK', 'eps_NQ', 'eps_NK', 'KmmInv')
        lhyp = T.dvector('lhyp')
        (M, K), N, Q = mu.shape, m.shape[0], Z.shape[1]
        s, sl2, sf2, l = T.exp(ls), T.exp(lhyp[0]), T.exp(lhyp[1]), T.exp(lhyp[2:2+Q])
        L = T.tril(lL - T.diag(T.diag(lL)) + T.diag(T.exp(T.diag(lL))))
        
        print 'Setting up cache...'
        Kmm = ker.RBF(sf2, l, Z) if not linear_model else ker.LIN(sl2, Z)
        KmmInv_cache = sT.matrix_inverse(Kmm)
        self.f_Kmm = theano.function([Z, lhyp], Kmm, name='Kmm')
        self.f_KmmInv = theano.function([Z, lhyp], KmmInv_cache, name='KmmInv_cache')
        self.update_KmmInv_cache()
        self.dKmm_d = {'Z': theano.function([Z, lhyp], T.jacobian(Kmm.flatten(), Z), name='dKmm_dZ'),
                       'lhyp': theano.function([Z, lhyp], T.jacobian(Kmm.flatten(), lhyp), name='dKmm_dlhyp')}

        print 'Setting up model...'
        if not self.use_hat:
            mu_scaled, L_scaled = sf2**0.5 * mu, sf2**0.5 * L
            X = m + s * eps_NQ
            U = mu_scaled + L_scaled.dot(eps_MK)
            Kmn = ker.RBF(sf2, l, Z, X) if not linear_model else ker.LIN(sl2, Z, X)
            Knn = ker.RBFnn(sf2, l, X) if not linear_model else ker.LINnn(sl2, X)
            A = KmmInv.dot(Kmn)
            B = Knn - T.sum(Kmn * KmmInv.dot(Kmn), 0)
            F = A.T.dot(U) + T.maximum(B, 1e-16)[:,None]**0.5 * eps_NK
            F = T.concatenate((T.zeros((N,1)), F), axis=1)
            S = T.nnet.softmax(F)
            LS = T.sum(T.log(T.maximum(T.sum(Y * S, 1), 1e-16)))
            if not linear_model:
                KL_U = -0.5 * (T.sum(KmmInv.T * T.sum(mu_scaled[:,None,:]*mu_scaled[None,:,:], 2))
                        + K * (T.sum(KmmInv.T * L_scaled.dot(L_scaled.T)) - M - 2.0*T.sum(T.log(T.diag(L_scaled)))
                               + 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm))))))
            else:
                KL_U = 0
            #KL_U = -0.5 * T.sum(T.sum(mu_scaled * KmmInv.dot(mu_scaled), 0) + T.sum(KmmInv * L_scaled.dot(L_scaled.T)) - M
            #                    - 2.0*T.sum(T.log(T.diag(L_scaled))) + 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm))))) if not linear_model else 0
        else:
            # mu_scaled, L_scaled = mu / sf2**0.5, L / sf2**0.5
            mu_scaled, L_scaled = mu / sf2, L / sf2
            X = m + s * eps_NQ
            U = mu_scaled + L_scaled.dot(eps_MK)
            Kmn = ker.RBF(sf2, l, Z, X) if not linear_model else ker.LIN(sl2, Z, X)
            Knn = ker.RBFnn(sf2, l, X) if not linear_model else ker.LINnn(sl2, X)
            B = Knn - T.sum(Kmn * KmmInv.dot(Kmn), 0)
            F = Kmn.T.dot(U) + T.maximum(B, 1e-16)[:,None]**0.5 * eps_NK
            F = T.concatenate((T.zeros((N,1)), F), axis=1)
            S = T.nnet.softmax(F)
            LS = T.sum(T.log(T.maximum(T.sum(Y * S, 1), 1e-16)))
            if not linear_model:
                KL_U = -0.5 * (T.sum(Kmm.T * T.sum(mu_scaled[:,None,:]*mu_scaled[None,:,:], 2))
                        + K * (T.sum(Kmm.T * L_scaled.dot(L_scaled.T)) - M - 2.0*T.sum(T.log(T.diag(L_scaled)))
                               - 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm))))))
            else:
                KL_U = 0

        KL_X_all = -0.5 * T.sum((m**2.0 + s**2.0)/sx2 - 1.0 - 2.0*ls + T.log(sx2), 1)
        KL_X = T.sum(KL_X_all)

        print 'Compiling...'
        inputs = {'Y': Y, 'Z': Z, 'm': m, 'ls': ls, 'mu': mu, 'lL': lL, 'lhyp': lhyp, 'KmmInv': KmmInv, 
            'eps_MK': eps_MK, 'eps_NQ': eps_NQ, 'eps_NK': eps_NK}
        z = 0.0*sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        f = zip(['X', 'U', 'S', 'LS', 'KL_U', 'KL_X', 'KL_X_all'], [X, U, S, LS, KL_U, KL_X, KL_X_all])
        self.f = {n: theano.function(inputs.values(), f+z, name=n, on_unused_input='ignore') for n,f in f}
        g = zip(['LS', 'KL_U', 'KL_X'], [LS, KL_U, KL_X])
        wrt = {'Z': Z, 'm': m, 'ls': ls, 'mu': mu, 'lL': lL, 'lhyp': lhyp, 'KmmInv': KmmInv}
        self.g = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
            on_unused_input='ignore') for gn,gv in g} for vn, vv in wrt.iteritems()}

        with open(model_file_name, 'wb') as file_handle:
            print 'Saving model...'
            sys.setrecursionlimit(2000)
            cPickle.dump([self.f, self.g, self.f_Kmm, self.f_KmmInv, self.dKmm_d], file_handle, protocol=cPickle.HIGHEST_PROTOCOL)
        
    def update_KmmInv_cache(self):
        self.KmmInv = self.f_KmmInv(self.params['Z'], self.params['lhyp']).astype(theano.config.floatX)

    def exec_f(self, f, Y = [[[0]]], modality = 0, mask = None):
        inputs, (M, K), (N, Q) = {}, self.params['mu' + str(modality)].shape, self.params['m'].shape
        inputs['Z'], inputs['m'], inputs['ls'] = self.params['Z'], self.params['m'], self.params['ls']
        inputs['mu'], inputs['lL'] = self.params['mu' + str(modality)], self.params['lL'][modality]
        inputs['lhyp'], inputs['KmmInv'] = self.params['lhyp'], self.KmmInv
        inputs['eps_MK'], inputs['eps_NQ'], inputs['eps_NK'] = np.random.randn(M,K), np.random.randn(N,Q), np.random.randn(N,K)
        inputs['Y'] = Y[modality] if len(Y) > 1 else Y[0]
        if mask is not None:
            inputs['Y'], inputs['m'], inputs['ls'] = inputs['Y'][~mask], inputs['m'][~mask], inputs['ls'][~mask]
            inputs['eps_NQ'], inputs['eps_NK'] = inputs['eps_NQ'][~mask], inputs['eps_NK'][~mask]
        return f(**inputs)

    def estimate(self, f, Y = [[[0]]], modality = 0, mask = None, samples = None):
        # np.random.seed(0)
        f_acc = np.array([self.exec_f(f, Y, modality, mask) for s in xrange(samples if samples != None else self.samples)])
        return np.nanmean(f_acc, 0), np.nanstd(f_acc, 0)

    def ELBO(self, Y, mask = None):
        if mask is None:
            mask = np.zeros((self.params['m'].shape[0], len(Y)), dtype=bool)
        ELBO, std_sum = self.exec_f(self.f['KL_X'], Y, mask=np.all(mask, 1)), 0
        for modality in xrange(len(Y)):
            LS, std = self.estimate(self.f['LS'], Y, modality, mask[:, modality])
            ELBO += self.exec_f(self.f['KL_U'], Y, modality, mask[:, modality]) + LS
            std_sum += std**2
        return ELBO, std_sum**0.5
