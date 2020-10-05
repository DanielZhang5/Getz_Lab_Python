# import time
import numpy as np
import math
import torch
from torch.distributions import StudentT

# for the user to decide device number
device_number = 0
if torch.cuda.is_available():
    torch.cuda.set_device(device_number)


# classes for sampling
class EpsiSampler:
    def __init__(self, x, epsi_nu):
        self.x = x
        self.len = self.x.shape[0]
        self.epsi_nu = epsi_nu
        self.tdistribution = StudentT(self.epsi_nu)

    def epsisamp(self, epsi, tau, mu):
        # assumes no covariance between epsilons; does not sample as a single block

        # Newton-Raphson iterations to find proposal density
        mu_f, hf, hf_inv = self.epsi_nr(epsi, mu, tau)

        # now propose with multivariate t centered at epsiMLE with covariance matrix from Hessian
        # note that since Hessian is diagonal, we can just simulate from n univariate t's.

        epsi_p = mu_f + hf_inv.neg().sqrt() * self.tdistribution.sample(torch.Size([self.len, 1]))
        # epsi_p = torch.randn(mu_f, -hf_inv)
        arat = self.pratepsi(epsi, epsi_p, tau, mu) + \
               tqrat(epsi, epsi_p, mu_f, mu_f, hf_inv.neg().sqrt(), hf_inv.neg().sqrt(), self.epsi_nu)

        ridx = torch.rand(self.len, 1).log() >= arat.clamp(max=0)
        ridx_float = ridx.type(torch.float32)

        epsi[~ridx] = epsi_p[~ridx]
        mrej = (1 - ridx_float).mean()
        return epsi, mrej

    # TODO: find out if .exp() legal here
    def pratepsi(self, epsi, epsi_p, tau, mu):
        pr = epsi_p * self.x / tau.sqrt() - (mu + epsi_p / tau.sqrt()).exp() - epsi_p ** 2 / 2 - \
             (epsi * self.x / tau.sqrt() - (mu + epsi / tau.sqrt()).exp() - epsi ** 2 / 2)
        return pr

    def epsi_nr(self, epsi, mu, tau):
        h, h_inv = 0, 0

        for i in range(1, 100):
            h, h_inv = self.hessepsi(epsi, tau, mu)

            # N - R update
            grad = self.gradepsi(epsi, tau, mu)
            epsi = epsi - h_inv * grad

            # we've reached a local maximum
            if grad.norm() < 1e-6:
                break
        return epsi, h, h_inv

    @staticmethod
    def hessepsi(epsi, tau, mu):
        h = -(mu + epsi / tau.sqrt()).exp() / tau - 1
        h_inv = 1 / h
        return h, h_inv

    def gradepsi(self, epsi, tau, mu):
        gr = self.x / tau.sqrt() - (mu + epsi / torch.sqrt(tau)).exp() / tau.sqrt() - epsi
        return gr


class MuTauSampler:
    def __init__(self, x, j, epsi, y=torch.tensor(0.), mutau_ihsf=torch.tensor(1.), t_mh_nr_iter=100,
                 taua=None, taub=None, mumu=None, mutau=None):
        self.x = x
        self.y = y
        self.reidx = torch.ones_like(x)  # subject to change i guess, kind of pointless to get a max from ones
        self.ncat = self.reidx.max()
        # for whatever reason, PyTorch always leaves a 0 in front for bin counts? Also, this just gives a scalar...
        # self.sums = torch.bincount(self.reidx, self.x)[1:]
        self.sums = self.x.sum()

        # previously set as properties of 'Y': should I organize this into a separate structure?
        self.j = j
        self.eps_x = epsi.view(1, -1) @ x  # usually has tidx slicing
        self.eBC = 0.
        self.d1 = torch.tensor(1)
        self.exp_eps_tau = torch.tensor(1)

        self.mutau_ihsf = mutau_ihsf
        self.t_mh_nr_iter = t_mh_nr_iter

        ncat = self.ncat.int().item()
        if taua is None:
            self.taua = torch.ones(ncat, 1)  # didn't add the + 1 yet
        else:
            self.taua = taua

        if taub is None:
            self.taub = torch.full([ncat, 1], 10.)
        else:
            self.taub = taub

        if mumu is None:
            self.mumu = torch.zeros(ncat, 1)
        else:
            self.mumu = mumu

        if mutau is None:
            self.mutau = torch.ones(ncat, 1)
        else:
            self.mutau = mutau

    def mutausamp(self, mu, tau, epsi):
        mutau = torch.tensor([[mu], [tau]])

        # Newton-Raphson iterations to find proposal density
        # note log transform of tau: tau -> exp(tau) in N-R iterations to allow it to range over all reals.
        mu_f, hf, hf_inv, is_max, use_ls = self.mutau_nr(mu, tau.log(), epsi)

        # now propose with multivariate Gaussian centered at MAP (tau log transformed) with covariance matrix from
        # Hessian
        evals, evecs = torch.eig(-hf_inv * self.mutau_ihsf, eigenvectors=True)
        # is mutau_p correctly set?
        mutau_p = mu_f + evecs @ torch.diag(torch.sqrt(evals[:, 0])) @ torch.rand((-hf_inv * self.mutau_ihsf).shape[0])

        # if we'd reached a local maximum, then parameterization of forward and reverse jumps identical.
        # if not, then reverse jump will have its own parameterization.

        # mu_r, hr, hr_inv = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        if not is_max:
            # way to limit the outputs in python?
            mu_r, hr, hr_inv, NOTHING1, NOTHING2 = self.mutau_nr(mu_f[0], mu_f[1], epsi)
        else:
            mu_r = mu_f
            hr, hr_inv = hf, hf_inv

        arat = self.pratmutau(mutau[0], mutau[1], mutau_p[0], mutau_p[1], epsi) + \
               mvnqrat(mutau, mutau_p, mu_f, mu_r, hf, hr, hf_inv, hr_inv, self.mutau_ihsf)

        # is the below line right? pycharm compiler not happy about the '.all()'
        if (torch.rand(1).log() < arat.clamp(max=0)).all():
            mu = mutau_p[0]
            tau = mutau_p[1].exp()
            rej = 0
        else:
            rej = 1

        return mu, tau, rej, use_ls

    def mutau_nr(self, mu, tau, epsi):
        is_max = False  # if we converged to a maximum
        # PyCharm compiler working weird for below line: says is_nr_bad is never used

        # initialized a bunch of variables
        # below 4 variables, according to pycharm compiler, are never used. do you need to initialize before while loop?
        is_nr_bad = False  # if regular N-R iterations are insufficient, so we need to employ linesearch
        h = torch.tensor(float('nan'))
        h_inv = torch.tensor(float('nan'))
        grad = torch.tensor(float('nan'))

        use_ls = False  # whether we used linesearch (for display purposes only)

        mu_prev = mu.clone()
        tau_prev = tau.clone()

        i = 1
        while True:
            tau_e = tau.exp()
            # should I make exp_eps_tau and d1 local variables that copy the value of the instance variables?
            # otherwise, we're changing the value of instance variables here (used to be values of Y)
            self.exp_eps_tau = torch.exp(epsi / tau_e.sqrt())
            self.d1 = self.exp_eps_tau  # mult Y.eBC

            grad = self.gradmutau(mu, tau_e, epsi)
            # would h and h.inverse need to be defined outside fo the while loop?
            h = self.hessmutau(mu, tau_e, epsi)
            h_inv = h.inverse()

            # change-of-variable chain rule factors
            h[1, 1] = grad[1] * tau_e + tau_e ** 2 * h[1, 1]
            h[0, 1] = tau_e * h[0, 1]
            h[1, 0] = h[0, 1]

            grad[1] *= tau_e

            # change-of-variable Jacobian factors
            grad[1] += 1

            h_np = h.numpy()
            eps = 2.2204e-16  # pytorch doesn't have eps

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
            print(grad.norm())

            if (grad.norm() <= 1e-6) and is_ndef:
                is_max = True
                break

            # 2. otherwise, employ line search
            fc_step = self.fcmutau(mu + step[0], tau + step[1], epsi)
            if is_nr_bad or fc_step.isnan().item() or fc_step.isinf().item() \
                    or (fc_step - self.fcmutau(tau, mu, epsi) < -1e-3):

                # indicate that we used linesearch for these iterations
                use_ls = True

                # 2.1. ensure N-R direction is even ascending. if not, use direction of gradient
                if not is_ndef:
                    step = grad

                # 2.2. regardless of the method, perform line search along direction of step
                s0 = step.norm()
                d_hat = step / s0
                # bound line search from below at current value
                fc = self.fcmutau(mu, tau, epsi) * torch.tensor([1, 1])

                # 2.3. do line search
                s = torch.tensor(0)
                for l in range(0, 50):
                    s = s0 * 0.5 ** (l - 1)

                    f = self.fcmutau(mu + s * d_hat[0], tau + s * d_hat[1], epsi)
                    # need to fix indexing?
                    if (fc[0][(l - 1) % 2] > fc[0][(l - 2) % 2]) and (fc[0][(l - 1) % 2] > f):
                        # correct indexing for python?
                        s = s0 * 0.5 ** (l - 1)
                        break
                    else:
                        fc[0][l % 2] = f

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

            print("iteration %s done" % i)

        mutau = torch.tensor([[mu], [tau]])

        return mutau, h, h_inv, is_max, use_ls

    def gradmutau(self, mu, tau, epsi):
        gr = torch.tensor([self.sums - mu.exp() * self.d1.sum() - self.mutau * tau * (mu - self.mumu),
                           (-self.eps_x + mu.exp() * (self.d1.view(1, -1) @ epsi)) / (2 * tau ** (3 / 2)) +
                           (2 * self.taua - 1) / (2 * tau) - 1 / self.taub - self.mutau / 2 * (mu - self.mumu) ** 2])
        return gr

    def hessmutau(self, mu, tau, epsi):
        h = torch.full([2, 2], float('nan'))

        h[0, 0] = -mu.exp() * self.d1.sum()
        h[1, 1] = 3 / 4 * tau ** (-5 / 2) * self.eps_x - \
                  mu.exp() * ((self.d1.view(1, -1) @ epsi) * (3 / 4) * tau ** (-5 / 2) +
                              (self.d1.view(1, -1) @ (epsi ** 2)) * (tau ** -3) / 4) - \
                  (2 * self.taua - 1) / (2 * tau ** 2)
        h[0, 1] = mu.exp() * (self.d1.view(1, -1) @ epsi) / (2 * tau ** (3 / 2)) - \
                  self.mutau * (mu - self.mumu)
        h[1, 0] = h[0, 1]

        return h

    # full conditional for mu, tau
    def fcmutau(self, mu, tau, epsi):
        tau_e = tau.exp()

        # torch.matmul((self.eBC*torch.ones_like(epsi)).view(1, -1), (epsi / tau_e.sqrt()).exp())
        # replace torch.ones_like(epsi) with Y.exp_eps_2_tau_2

        fc = mu * self.sums + self.eps_x / tau_e.sqrt() - mu.exp() * \
             (self.eBC * torch.ones_like(epsi).view(1, -1) @ (epsi / tau_e.sqrt()).exp()) + \
             normgampdf(mu, tau_e, self.taua, self.taub, self.mumu, self.mutau, True) + tau

        return fc

    # log posterior ratio for mu,tau (tau log transformed)
    def pratmutau(self, mu, tau, mu_p, tau_p, epsi):
        tau_p_e, tau_e = tau_p.exp(), tau.exp()  # necessary?

        pr = self.fcmutau(mu_p, tau_p, epsi) - \
             self.fcmutau(mu, tau, epsi)

        return pr


def tqrat(th_0, th_p, mu_f, mu_r, sig_f, sig_r, nu):
    qrat = -torch.log(sig_r) - (nu + 1) / 2 * torch.log(1 + (th_0 - mu_r) ** 2 / (nu * sig_r ** 2)) - \
           (-torch.log(sig_f) - (nu + 1) / 2 * torch.log(1 + (th_p - mu_f) ** 2 / (nu * sig_f ** 2)))
    return qrat


# mutau, mutau_p, mu_f, mu_r, hf, hr, hf_inv, hr_inv, self.mutau_ihsf
# what should these dimensions be?
def mvnqrat(th0, thP, muF, muR, hessF, hessR, hessFinv, hessRinv, ihsf):
    qrat = (-torch.log(torch.det(-hessRinv / ihsf)) + (th0 - muR).view(1, -1) @ hessR * ihsf @ (th0 - muR) -
            (-torch.log(torch.det(-hessFinv / ihsf)) + (thP - muF).transpose(0, 1) @ hessF * ihsf @ (thP - muF))) / 2

    # (-(-hessRinv / ihsf).det().log()) + \
    # (th0 - muR).view(1, -1) @ hessR * ihsf * (th0 - muR) - \
    # (-(-hessFinv / ihsf).det().log()) + \
    # (thP - muF).view(1, -1) @ hessF * ihsf * (thP - muF) / 2

    return qrat


# is there a pytorch built-in function for this?
# should I put into separate file?
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


poisson = torch.poisson(torch.exp(-3. + 0.9 * torch.randn(5000, 1)))
X = MuTauSampler(poisson, 1, torch.zeros_like(poisson), mumu=torch.tensor([[-3.]]), taua=torch.tensor([[10.]]),
                 taub=torch.tensor([[0.2]]))

epsi_test = torch.zeros_like(X.x)
tau_test = torch.tensor(1.)  # 1.234567
mu_test = torch.tensor(1.)  # -3.

# mu_result, tau_result, rej_result, use_ls_result = X.mutausamp(mu_test, tau_test, epsi_test)
# print("mu: %s\ntau: %s\nrej: %s\n use_ls: %s" % (mu_result, tau_result, rej_result, use_ls_result))

mutau_result, h_result, h_inv_result, is_max_result, use_ls_result1 = X.mutau_nr(mu_test, tau_test, epsi_test)
print("mutau: %s\nh: %s\nh_inv: %s\nis_max: %s\nuse_ls: %s" % (mutau_result, h_result, h_inv_result,
                                                               is_max_result, use_ls_result1))
