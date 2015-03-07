import CLGP_theano_model
from CLGP_theano_model import CLGP_model
import numpy as np

class CLGP_opt:
    def __init__(self, params, Y, mask = None, sx2 = 1, linear_model = False, samples = 20, use_hat = False):
        self.Y = Y
        if mask is None:
            self.mask = np.zeros((Y[0].shape[0], len(Y)), dtype=bool)
        else:
            self.mask = mask
        self.clgp = CLGP_model(params, sx2, linear_model=linear_model, samples=samples, use_hat=use_hat)
        self.ELBO, self.f = self.clgp.ELBO, self.clgp.f
        self.params, self.KmmInv = self.clgp.params, self.clgp.KmmInv
        self.exec_f, self.estimate = self.clgp.exec_f, self.clgp.estimate
        self.param_updates = {n: np.zeros_like(v) for n, v in params.iteritems()}
        self.moving_mean_squared = {n: np.zeros_like(v) for n, v in params.iteritems()}
        self.learning_rates = {n: 1e-2*np.ones_like(v) for n, v in params.iteritems()}

    def get_KmmInv_grad(self, Y, mask, (M, Q)):
        dKL_U_dKmmInv, dLS_dKmmInv = {}, {}
        for modality in xrange(len(Y)):
            dKL_U_dKmmInv[modality] = self.exec_f(self.clgp.g['KmmInv']['KL_U'], Y, modality, mask[:, modality])
            dLS_dKmmInv[modality] = self.estimate(self.clgp.g['KmmInv']['LS'], Y, modality, mask[:, modality])[0]
        
        df_dn_i = {'Z': {'KL_U': {}, 'LS': {}}, 'lhyp': {'KL_U': {}, 'LS': {}}}
        dKmm_dlhyp = self.clgp.dKmm_d['lhyp'](self.params['Z'], self.params['lhyp']).reshape(M, M, -1)
        dKmm_dn_KmmInv = np.dot(dKmm_dlhyp.transpose((2, 0, 1)), self.clgp.KmmInv)
        KmmInv_dKmm_dn_KmmInv = np.dot(dKmm_dn_KmmInv.transpose((0,2,1)),self.clgp.KmmInv.T).transpose((0,2,1))
        dKmmInv_dlhyp = -1.0 * KmmInv_dKmm_dn_KmmInv.transpose((1,2,0))

        dKmm_dZ = self.clgp.dKmm_d['Z'](self.params['Z'], self.params['lhyp']).reshape(M, M, M, Q)
        dKmm_dn_KmmInv = np.dot(dKmm_dZ.transpose((2,3,0,1)), self.clgp.KmmInv)
        KmmInv_dKmm_dn_KmmInv = np.dot(dKmm_dn_KmmInv.transpose((0,1,3,2)),self.clgp.KmmInv.T).transpose((0,1,3,2))
        dKmmInv_dZ = -1.0 * KmmInv_dKmm_dn_KmmInv.transpose((2,3,0,1))

        for modality in xrange(len(Y)):
            df_dn_i['lhyp']['KL_U'][modality] = (dKL_U_dKmmInv[modality][:,:,None] * dKmmInv_dlhyp).sum(0).sum(0)
            df_dn_i['lhyp']['LS'][modality] = (dLS_dKmmInv[modality][:,:,None] * dKmmInv_dlhyp).sum(0).sum(0)
            df_dn_i['Z']['KL_U'][modality] = (dKL_U_dKmmInv[modality][:,:,None,None] * dKmmInv_dZ).sum(0).sum(0)
            df_dn_i['Z']['LS'][modality] = (dLS_dKmmInv[modality][:,:,None,None] * dKmmInv_dZ).sum(0).sum(0)
        return df_dn_i

    def get_grad(self, param_name, Y, KmmInv_grad, mask = None):
        if 'mu' in param_name:
            modality = int(param_name[2:])
            return (self.exec_f(self.clgp.g['mu']['KL_U'], Y, modality, mask[:, modality]) 
                     + self.estimate(self.clgp.g['mu']['LS'], Y, modality, mask[:, modality])[0])
        grad = []
        for modality in xrange(len(Y)):
            m = mask[:, modality]
            if param_name in ['m', 'ls']:
                g = np.zeros_like(self.clgp.params[param_name])
                g[~m] = (self.exec_f(self.clgp.g[param_name]['KL_U'], Y, modality, m) 
                    + self.estimate(self.clgp.g[param_name]['LS'], Y, modality, m)[0])
                grad += [g]
            else:
                # grad += [(self.exec_f(self.clgp.g[param_name]['KL_U'], Y, modality, m)
                #     + self.estimate(self.clgp.g[param_name]['LS'], Y, modality, m)[0])]
                grad_ls, grad_std = self.estimate(self.clgp.g[param_name]['LS'], Y, modality, m)
                grad += [self.exec_f(self.clgp.g[param_name]['KL_U'], Y, modality, m) + grad_ls]
                if param_name in ['Z', 'lhyp']:
                    grad[modality] += (KmmInv_grad[param_name]['KL_U'][modality] 
                        + KmmInv_grad[param_name]['LS'][modality])
        if param_name in ['Z', 'lhyp', 'm', 'ls']:
            grad = np.sum(grad, 0)
        if param_name in ['m', 'ls']:
            m = ~np.any(~mask, axis=1)
            grad[~m] += self.exec_f(self.clgp.g[param_name]['KL_X'], Y, mask=m)

        # DEBUG
        if param_name == 'lhyp' and np.any(np.abs(grad) < grad_std / np.sqrt(self.clgp.samples)):
                #print 'Large noise, recomputing. lhyp grad mean:', grad, ', std:', grad_std / np.sqrt(self.clgp.samples)

                samples = self.clgp.samples * 10
                grad = []
                for modality in xrange(len(Y)):
                    m = mask[:, modality]
                    grad_ls, grad_std = self.estimate(self.clgp.g[param_name]['LS'], Y, modality, m, samples=samples)
                    grad += [self.exec_f(self.clgp.g[param_name]['KL_U'], Y, modality, m) + grad_ls]
                    grad[modality] += (KmmInv_grad[param_name]['KL_U'][modality]
                        + KmmInv_grad[param_name]['LS'][modality])
                grad = np.sum(grad, 0)

                self.grad_std = grad_std

        return np.array(grad)

    def opt_one_step(self, params, iteration, opt = 'rmsprop', learning_rate_adapt = 0.2, use_einsum = True):
        KmmInv_grad = self.get_KmmInv_grad(self.Y, self.mask, self.params['Z'].shape)
        
        for param_name in params:
            # DEBUG
            if opt == 'grad_ascent' or param_name in ['ls']:
                self.grad_ascent_one_step(param_name, [param_name, self.Y, KmmInv_grad, self.mask], 
                    learning_rate_decay = learning_rate_adapt * 100 / (iteration + 100.0))
            elif opt == 'rmsprop':
                self.rmsprop_one_step(param_name, [param_name, self.Y, KmmInv_grad, self.mask], 
                    learning_rate_adapt = learning_rate_adapt)#, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))
            if param_name in ['lhyp']:
                self.params[param_name] = np.clip(self.params[param_name], -8, 8)
            if param_name in ['lhyp', 'Z']:
                self.clgp.update_KmmInv_cache()

    def grad_ascent_one_step(self, param_name, grad_args, momentum = 0.9, learning_rate_decay = 1):
        self.clgp.params[param_name] += (learning_rate_decay*self.learning_rates[param_name]* self.param_updates[param_name])
        grad = self.get_grad(*grad_args)
        if param_name in ['lhyp']:
            self.param_updates[param_name] = momentum*self.param_updates[param_name] + (1. - momentum)*grad
        else:
            self.param_updates[param_name] = grad

    def rmsprop_one_step(self, param_name, grad_args, decay = 0.9, momentum = 0, learning_rate_adapt = 0.05, 
        learning_rate_min = 1e-6, learning_rate_max = 10):
        # RMSPROP: Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        # Implementation based on https://github.com/BRML/climin/blob/master/climin/rmsprop.py
        
        # We use Nesterov momentum: first, we make a step according to the momentum and then we calculate the gradient.
        step1 = self.param_updates[param_name] * momentum
        self.params[param_name] += step1
        grad = self.get_grad(*grad_args)

        self.moving_mean_squared[param_name] = (decay * self.moving_mean_squared[param_name] + (1 - decay) * grad ** 2)
        step2 = self.learning_rates[param_name] * grad / (self.moving_mean_squared[param_name] + 1e-8)**0.5

        # DEBUG
        if param_name == 'lhyp':
            step2 = np.clip(step2, -0.1, 0.1)

        self.params[param_name] += step2

        step = step1 + step2

        # Step rate adaption. If the current step and the momentum agree, we slightly increase the step rate for that dimension.
        if learning_rate_adapt:
            # This code might look weird, but it makes it work with both numpy and gnumpy.
            step_non_negative = step > 0
            step_before_non_negative = self.param_updates[param_name] > 0
            agree = (step_non_negative == step_before_non_negative) * 1.
            adapt = 1 + agree * learning_rate_adapt * 2 - learning_rate_adapt
            self.learning_rates[param_name] *= adapt
            self.learning_rates[param_name] = np.clip(self.learning_rates[param_name], learning_rate_min, learning_rate_max)

        self.param_updates[param_name] = step

    def choose_best_z(self, ind, Y_true, mask, samples=20):
        """
        Assign m[i] to the best location among all the inducing points.
        """
        orig_params = {'m': self.params['m'], 'ls': self.params['ls']}
        N = len(ind)
        M = self.params['Z'].shape[0]

        self.params['ls'] = self.params['ls'][ind]
        f = np.zeros((M + 1, N))
        for m in xrange(M + 1):
            if m < M:
                self.params['m'] = np.tile(self.params['Z'][m], (N, 1))
            else:
                self.params['m'] = orig_params['m'][ind]

            # KL.
            kl_x = self.exec_f(self.f['KL_X_all'])
            f[m] += kl_x

            # Likelihood.
            for modality in xrange(len(Y_true)):
                S, _ = self.estimate(self.f['S'], modality=modality, samples=samples)

                Y_ind = Y_true[modality][ind]
                mask_ind = mask[:, modality][ind]
                f[m] += np.log(np.maximum(np.sum(S * Y_ind, 1), 1e-16)) * mask_ind

        self.params['m'], self.params['ls'] = orig_params['m'], orig_params['ls']

        best_z = np.argmax(f, 0)

        # Do not change m if best_z == M.
        self.params['m'][ind[best_z < M]] = self.params['Z'][best_z[best_z < M]]

        return best_z
