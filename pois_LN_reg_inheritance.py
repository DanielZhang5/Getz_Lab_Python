import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import StudentT

# for the user to decide device number
device_number = 0
if torch.cuda.is_available():
    torch.cuda.set_device(device_number)

torch.set_default_dtype(torch.float64)


# wrapper: ch, chfield and basechange


class Sampler:
    def __init__(self, x):
        self.x = x


class SamplerWithPAndY(Sampler):
    def __init__(self, x, t_mh_nr_iter, taua, taub):
        Sampler.__init__(self, x)

        self.reidx = torch.ones_like(x)
        self.ncat = self.reidx.max()
        self.sums = self.x.sum()
        # self.sums = torch.bincount(self.reidx, self.x)[1:]

        # '''
        # previously set as properties of 'Y'
        self.j = None
        self.eps_x = None  # epsi.view(1, -1) @ x  # usually has tidx slicing
        self.eBC = torch.tensor(1.)
        self.exp_eps_tau = None
        self.d1 = None
        # '''

        self.t_mh_nr_iter = 100 if t_mh_nr_iter is None else t_mh_nr_iter
        self.ncat_index = self.ncat.int().item()
        self.taua = torch.ones(self.ncat_index + 1, 1) if taua is None else taua
        self.taub = torch.full([self.ncat_index, 1], 10.) if taub is None else taub


class EpsiSampler(Sampler):
    def __init__(self, x, epsi_nu):
        Sampler.__init__(self, x)
        self.len = self.x.shape[0]
        self.epsi_nu = epsi_nu
        self.tdistribution = StudentT(self.epsi_nu)

    def epsisamp(self, epsi, tau, mu):
        # assumes no covariance between epsilons; does not sample as a single block

        # Newton-Raphson iterations to find proposal density
        mu_f, hf, hf_inv = self.__epsi_nr(epsi, mu, tau)

        # now propose with multivariate t centered at epsiMLE with covariance matrix from Hessian
        # note that since Hessian is diagonal, we can just simulate from n univariate t's.

        epsi_p = mu_f + hf_inv.neg().sqrt() * self.tdistribution.sample(torch.Size([self.len, 1]))
        # epsi_p = torch.randn(mu_f, -hf_inv)
        arat = self.__pratepsi(epsi, epsi_p, tau, mu) + \
               tqrat(epsi, epsi_p, mu_f, mu_f, hf_inv.neg().sqrt(), hf_inv.neg().sqrt(), self.epsi_nu)

        ridx = torch.rand(self.len, 1).log() >= arat.clamp(max=0)
        ridx_float = ridx.type(torch.float32)

        epsi[~ridx] = epsi_p[~ridx]
        mrej = (1 - ridx_float).mean()
        return epsi, mrej

    def __pratepsi(self, epsi, epsi_p, tau, mu):
        pr = epsi_p * self.x / tau.sqrt() - (mu + epsi_p / tau.sqrt()).exp() - epsi_p ** 2 / 2 - \
             (epsi * self.x / tau.sqrt() - (mu + epsi / tau.sqrt()).exp() - epsi ** 2 / 2)
        return pr

    def __epsi_nr(self, epsi, mu, tau):
        h, h_inv = 0, 0

        for i in range(1, 100):
            h, h_inv = self.__hessepsi(epsi, tau, mu)

            # N - R update
            grad = self.__gradepsi(epsi, tau, mu)
            epsi = epsi - h_inv * grad

            # we've reached a local maximum
            if grad.norm() < 1e-6:
                break
        return epsi, h, h_inv

    @staticmethod
    def __hessepsi(epsi, tau, mu):
        h = -(mu + epsi / tau.sqrt()).exp() / tau - 1
        h_inv = 1 / h
        return h, h_inv

    def __gradepsi(self, epsi, tau, mu):
        gr = self.x / tau.sqrt() - (mu + epsi / torch.sqrt(tau)).exp() / tau.sqrt() - epsi
        return gr


class MuTauSampler(SamplerWithPAndY):
    def __init__(self, x, t_mh_nr_iter=None, taua=None, taub=None,
                 mutau_ihsf=torch.tensor(1.), mumu=None, mutau=None):
        SamplerWithPAndY.__init__(self, x, t_mh_nr_iter, taua, taub)

        # previously properties of 'X.P'
        self.mutau_ihsf = mutau_ihsf
        self.mumu = torch.zeros(self.ncat_index, 1) if mumu is None else mumu
        self.mutau = torch.ones(self.ncat_index, 1) if mutau is None else mutau

    def mutausamp(self, mu, tau, epsi, j=0):
        # select jth mu/tau from array
        mu, tau = mu[j], tau[j]

        # constant variables
        self.j = j
        self.eps_x = epsi.view(1, -1) @ self.x

        mutau = torch.tensor([[mu], [tau.log()]])

        # Newton-Raphson iterations to find proposal density
        # note log transform of tau: tau -> exp(tau) in N-R iterations to allow it to range over all reals.
        mu_f, hf, hf_inv, is_max, use_ls = self.__mutau_nr(mu, tau.log(), epsi)

        # now propose with multivariate Gaussian centered at MAP (tau log transformed) with covariance matrix from
        # Hessian
        evals, evecs = torch.eig(-hf_inv * self.mutau_ihsf, eigenvectors=True)
        mutau_p = mu_f + (evecs @ torch.diag(torch.sqrt(evals[:, 0])) @
                          torch.rand((-hf_inv * self.mutau_ihsf).shape[0])).view(-1, 1)

        # if we'd reached a local maximum, then parameterization of forward and reverse jumps identical.
        # if not, then reverse jump will have its own parameterization.
        if not is_max:
            # TODO: way to limit the outputs in python?
            mu_r, hr, hr_inv = self.__mutau_nr(mu_f[0], mu_f[1], epsi)[0:2]
        else:
            mu_r = mu_f
            hr, hr_inv = hf, hf_inv

        arat = self.__pratmutau(mutau[0], mutau[1], mutau_p[0], mutau_p[1], epsi) + \
               mvnqrat(mutau, mutau_p, mu_f, mu_r, hf, hr, hf_inv, hr_inv, self.mutau_ihsf)

        # is the below line right? pycharm compiler not happy about the '.all()'
        if (torch.rand(1).log() < arat.clamp(max=0)).all():
            mu = mutau_p[0]
            tau = mutau_p[1].exp()
            rej = 0
        else:
            rej = 1

        return mu, tau, rej, use_ls

    def __mutau_nr(self, mu, tau, epsi):
        is_max = False  # if we converged to a maximum
        is_nr_bad = False  # if regular N-R iterations are insufficient, so we need to employ line search

        use_ls = False  # whether we used line search (for display purposes only)

        mu_prev = mu.clone()
        tau_prev = tau.clone()

        i = 1
        while True:
            tau_e = tau.exp()
            # should I make exp_eps_tau and d1 local variables that copy the value of the instance variables?
            # otherwise, we're changing the value of instance variables here (used to be values of Y)
            self.exp_eps_tau = (epsi / tau_e.sqrt()).exp()
            self.d1 = self.exp_eps_tau  # mult Y.eBC

            grad = self.__gradmutau(mu, tau_e, epsi).view(-1, 1)
            h = self.__hessmutau(mu, tau_e, epsi)
            h_inv = h.inverse()

            # change-of-variable chain rule factors
            h[1, 1] = grad[1] * tau_e + tau_e ** 2 * h[1, 1]
            h[0, 1] = tau_e * h[0, 1]
            h[1, 0] = h[0, 1]
            grad[1] *= tau_e

            # change-of-variable Jacobian factors
            grad[1] += 1

            h_np = h.numpy()
            eps = 2.2204e-16

            # if Hessian is problematic, rewind an iteration and use line search by default
            if (1 / np.linalg.cond(h_np, 1) < eps) or h[:].isnan().any().item() or h[:].isinf().any().item():
                # if we're in a problematic region at the first iteration, break and hope for the best
                if i == 1:
                    h = torch.eye(2)
                    h_inv = -torch.eye(2)
                    print("WARNING: Full conditional shifted significantly since last iteration!")

                    break

                is_nr_bad = True
                i -= 1

                mu = mu_prev.clone()
                tau = tau_prev.clone()

                continue
            else:
                is_nr_bad = False

            # check if Hessian is negative definite
            is_ndef = (not is_nr_bad) and (h.eig()[0][:, 0] < 0).all().item()

            # if Newton-Raphson will work fine, use it
            h_inv = h.inverse()
            step = -h_inv @ grad

            # check if we've reached a local maximum

            if (grad.norm() <= 1e-6) and is_ndef:
                is_max = True
                break

            # 2. otherwise, employ line search
            fc_step = self.__fcmutau(mu + step[0], tau + step[1], epsi)
            if is_nr_bad or fc_step.isnan().item() or fc_step.isinf().item() \
                    or (fc_step - self.__fcmutau(mu, tau, epsi) < -1e-3):

                # indicate that we used line search for these iterations
                use_ls = True

                # 2.1. ensure N-R direction is even ascending. if not, use direction of gradient
                if not is_ndef:
                    step = grad

                # 2.2. regardless of the method, perform line search along direction of step
                s0 = step.norm()
                d_hat = step / s0

                # bound line search from below at current value
                fc = self.__fcmutau(mu, tau, epsi) * torch.tensor([1, 1])

                # 2.3. do line search
                s = torch.tensor(0)
                for ls in range(0, 50):
                    s = s0 * 0.5 ** (ls - 1)

                    f = self.__fcmutau(mu + s * d_hat[0], tau + s * d_hat[1], epsi)
                    # need to fix indexing?
                    if (fc[0][(ls - 1) % 2] > fc[0][(ls - 2) % 2]) and (fc[0][(ls - 1) % 2] > f):
                        # correct indexing for python?
                        s = s0 * 0.5 ** (ls - 1)
                        break
                    else:
                        fc[0][ls % 2] = f

                step = d_hat * s

            # update mu, tau
            mu_prev, tau_prev = mu.clone(), tau.clone()
            mu, tau = mu + step[0], tau + step[1]

            i += 1
            if i > self.t_mh_nr_iter:
                if not is_ndef:
                    h = torch.eye(2)
                    print('WARNING: Newton-Raphson terminated at non-concave point!')
                break

        mutau = torch.tensor([[mu], [tau]])

        return mutau, h, h_inv, is_max, use_ls

    def __gradmutau(self, mu, tau, epsi):
        # TODO: When we add covariates, add Y.j indexing to self.sums
        gr = torch.tensor([self.sums - mu.exp() * self.d1.sum() - self.mutau[self.j] * tau * (mu - self.mumu[self.j]),
                           (-self.eps_x + mu.exp() * (self.d1.view(1, -1) @ epsi)) / (2 * tau ** (3 / 2)) +
                           (2 * self.taua[self.j] - 1) / (2 * tau) - 1 / self.taub[self.j] - self.mutau[self.j] / 2 *
                           (mu - self.mumu[self.j]) ** 2])

        return gr

    def __hessmutau(self, mu, tau, epsi):
        h = torch.full([2, 2], float('nan'))

        h[0, 0] = -mu.exp() * self.d1.sum() - self.mutau[self.j] * tau
        h[1, 1] = 3 / 4 * tau ** (-5 / 2) * self.eps_x - \
                  mu.exp() * ((self.d1.view(1, -1) @ epsi) * (3 / 4) * tau ** (-5 / 2) +
                              (self.d1.view(1, -1) @ (epsi ** 2)) * (tau ** -3) / 4) - \
                  (2 * self.taua[self.j] - 1) / (2 * tau ** 2)
        h[0, 1] = mu.exp() * (self.d1.view(1, -1) @ epsi) / (2 * tau ** (3 / 2)) - \
                  self.mutau[self.j] * (mu - self.mumu)
        h[1, 0] = h[0, 1]

        return h

    # full conditional for mu, tau
    def __fcmutau(self, mu, tau, epsi):
        tau_e = tau.exp()

        # replace self.reidx with Y.exp_eps_2_tau_2
        # TODO: When we add covariates, add Y.j indexing to self.sums
        fc = mu * self.sums + self.eps_x / tau_e.sqrt() - mu.exp() * \
             (self.eBC * self.reidx.view(1, -1) @ (epsi / tau_e.sqrt()).exp()) + \
             normgampdf(mu, tau_e, self.taua[self.j], self.taub[self.j], self.mumu[self.j], self.mutau[self.j], True) \
             + tau

        return fc

    # log posterior ratio for mu,tau (tau log transformed)
    def __pratmutau(self, mu, tau, mu_p, tau_p, epsi):
        # tau_p_e, tau_e = tau_p.exp(), tau.exp()  # necessary?

        pr = self.__fcmutau(mu_p, tau_p, epsi) - \
             self.__fcmutau(mu, tau, epsi)

        return pr


class TauSampler(SamplerWithPAndY):
    def __init__(self, x, t_mh_nr_iter=None, taua=None, taub=None):
        SamplerWithPAndY.__init__(self, x, t_mh_nr_iter, taua, taub)

    def tausamp(self, tau, mu, epsi):
        pass

    # log posterior ration for tau0 (tau0 log transformed)
    def __prattau(self, tau, tau_p, mu, epsi):
        tau_p_e = tau_p.exp()
        tau_e = tau.exp()

        pr = self.eps_x / tau_p_e.sqrt() - (self.eBC * self.reidx.view(1, -1)) @ (mu.exp() *
                                                                                  (epsi / tau_p_e.sqrt()).exp())

        return pr

    # first derivative of log p(tau0|-)
    def __gradtau(self, tau, mu, epsi):
        pass

    # Newton-Raphson iterations for tau\
    def __tau_nr(self, tau, mu, epsi):
        pass


def tqrat(th_0, th_p, mu_f, mu_r, sig_f, sig_r, nu):
    qrat = -sig_r.log() - (nu + 1) / 2 * (1 + (th_0 - mu_r) ** 2 / (nu * sig_r ** 2)).log() - \
           (-sig_f.log() - (nu + 1) / 2 * (1 + (th_p - mu_f) ** 2 / (nu * sig_f ** 2)).log())
    return qrat

    # -torch.log(sig_r) - (nu + 1) / 2 * torch.log(1 + (th_0 - mu_r) ** 2 / (nu * sig_r ** 2)) - \
    # (-torch.log(sig_f) - (nu + 1) / 2 * torch.log(1 + (th_p - mu_f) ** 2 / (nu * sig_f ** 2)))


def mvnqrat(th_0, th_p, mu_f, mu_r, hess_f, hessR, hess_f_inv, hess_r_inv, ihsf):
    qrat = (-(-hess_r_inv / ihsf).det().log() + (th_0 - mu_r).view(1, -1) @ hessR * ihsf @ (th_0 - mu_r) -
           (-(-hess_f_inv / ihsf).det().log() + (th_p - mu_f).transpose(0, 1) @ hess_f * ihsf @ (th_p - mu_f))) / 2
    return qrat

    # (-torch.log(torch.det(-hess_r_inv / ihsf)) + (th_0 - mu_r).view(1, -1) @ hessR * ihsf @ (th_0 - mu_r) -
    # (-torch.log(torch.det(-hess_f_inv / ihsf)) + (th_p - mu_f).transpose(0, 1) @ hess_f * ihsf @ (th_p - mu_f))) / 2


# put into separate file?
def normgampdf(mu, tau, a, b, m, t, logmode=False):
    # argument checks
    if not (mu.shape == tau.shape):
        raise Exception('Size of mu and tau must be the same size (mu is %s, tau is %s)' % (mu.shape, tau.shape))
    if not (a.shape == b.shape == m.shape == t.shape):
        raise Exception('Size of a/b/m/t parameters must be the same size (a: %s, b: %s, m: %s, t: %s)' %
                        (a.shape, b.shape, m.shape, t.shape))

    pix2 = torch.tensor(2. * math.pi)
    if not logmode:
        p = tau ** (a - 1) * (-tau / b).exp() * (-t * tau / 2 * (mu - m) ** 2).exp() * tau.sqrt()
        z = b ** a * pix2.sqrt() * a.lgamma().exp() / t.sqrt()

        p = p / z
    else:
        p = (a - 1) * tau.log() - tau / b - t * tau / 2 * (mu - m) ** 2 + tau.log() / 2
        z = a * b.log() + pix2.sqrt().log() + a.lgamma() - t.log() / 2

        p = p - z

    return p


poisson = torch.poisson(torch.exp(-3. + 0.9 * torch.randn(50000, 1)))
# hyperparameters must be a 1x1 matrix
X = MuTauSampler(poisson)  # , mumu=torch.tensor([[-3.]]), taua=torch.tensor([[10.]]), taub=torch.tensor([[.2]]))

epsi_test = torch.zeros_like(X.x)
tau_test = torch.tensor([1.])  # 1.234567
mu_test = torch.tensor([1.])  # -3.
j_test = 0

mu_array = np.array([])
tau_array = np.array([])

# plt.plot([1, 2, 3], [1, 2, 3], "bo", markersize=2)
# plt.show()

for i in range(1, 2000):
    mu_result, tau_result, _, _ = X.mutausamp(mu_test, tau_test, epsi_test)
    mu_array = np.append(mu_array, mu_result)
    tau_array = np.append(tau_array, 1/tau_result.sqrt())

    # print(mu_array)
    print("points: %s" % i)

print(mu_array[0])
print(tau_array[0])
plt.xlim(-5, 0)
plt.ylim(0, 5)
plt.plot(mu_array, tau_array, 'ko', markersize=2)
plt.plot(-3, 0.9, 'ro', markersize=2)
plt.xlabel("mu")
plt.ylabel("tau")
plt.show()
