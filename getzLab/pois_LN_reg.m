X = [];
X.reidx = ones(size(X.x));
X.ncat = max(X.reidx);
X.sums = accumarray(X.reidx, X.x);
X.reidx_bin = sparse(1:length(X.reidx), X.reidx, 1);
X.y = 0



function qrat = mvnqrat(th0, thP, muF, muR, hessF, hessR, hessFinv, hessRinv, ihsf)
  qrat = (-log(det(-hessRinv/ihsf)) + (th0 - muR)'*hessR*ihsf*(th0 - muR) - ...
	 (-log(det(-hessFinv/ihsf)) + (thP - muF)'*hessF*ihsf*(thP - muF)))/2;
end



%a M-H sample from p(mu_j, tau_j|-)
function [mu, tau, rej, use_LS] = mutausamp(mu, tau, epsi, j, X),
  %select jth mu/tau from array
  mu = mu;
  tau = tau;
  %TODO: select proper hyperparameters in the same fashion

  %tidx = logical(X.reidx_bin(:, j));

  %constant variables
  Y = [];
  Y.j = j;
  Y.eps_x = epsi'*X.x(tidx);
  # if max(size(X.y)) > 1, y = X.y(tidx); else, y = X.y; end
  Y.eBC = 1; %exp(X.c(tidx, :)* + y); %XXX: does this only work if X.y is a scalar?
  
  % Y.exp_eps_2_ = exp(/sqrt);

  mutau = [mu; log(tau)];

  %Newton-Raphson iterations to find proposal density
  %note log transform of tau: tau -> exp(tau) in N-R iterations to allow it to range over all reals.
  [muF, HF, HFinv, is_max, use_LS] = mutauNR(mu, log(tau), epsi, X, Y);

  %now propose with multivariate Gaussian centered at MAP (tau log transformed) with covariance matrix from Hessian
  mutauP = mvnrnd(muF, -HFinv*X.P.mutau_ihsf)';

  %if we'd reached a local maximum, then parameterization of forward and reverse jumps identical.
  %if not, then reverse jump will have its own parameterization.
  if ~is_max,
    [muR, HR, HRinv] = mutauNR(muF(1), muF(2), epsi, X, Y);
  else
    muR = muF;
    HR = HF; HRinv = HFinv;
  end

  arat = pratmutau(mutau(1), mutau(2), mutauP(1), mutauP(2), epsi, X, Y) + ...
	 mvnqrat(mutau, mutauP, muF, muR, HF, HR, HFinv, HRinv, X.P.mutau_ihsf);

  if log(rand) < min(0, arat),
    mu = mutauP(1);
    tau = exp(mutauP(2));
    rej = 0;
  else
    rej = 1;
  end
end

%full conditional for mu,tau
function fc = fcmutau(mu, tau, epsi, X, Y)
  tau_e = exp(tau);

  fc = mu*X.sums + Y.eps_x/sqrt(tau_e) - exp(mu)*(Y.eBC)'*exp(epsi/sqrt(tau_e))) + ...
        normgampdf(mu, tau_e, X.P.taua, X.P.taub, X.P.mumu, X.P.mutau, true) + tau;
end

%log posterior ratio for mu,tau (tau log transformed)
function pr = pratmutau(mu, tau, muP, tauP, epsi, X, Y)
  tauP_e = exp(tauP); tau_e = exp(tau);

  pr = fcmutau(muP, tauP, epsi, X, Y) - ...
       fcmutau(mu, tau, epsi, X, Y);
end

%gradient of log p(mu, tau|-)
function gr = gradmutau(mu, tau, , epsi, X, Y)
  gr = [X.sums - exp(mu)*sum(Y.d1) - ...
        X.P.mutau*tau*(mu - X.P.mumu), ...
        (-Y.eps_x + exp(mu)*(Y.d1'*epsi))/(2*tau^(3/2)) + ...
         (2*X.P.taua - 1)/(2*tau) - 1/X.P.taub - X.P.mutau/2*(mu - X.P.mumu)^2];
end

%Hessian of log p(mu, tau|-)
function H = hessmutau(mu, tau, , epsi, X, Y)
  H = NaN(2);

  H(1, 1) = -exp(mu)*sum(Y.d1) - ...
            X.P.mutau*tau;
  H(2, 2) = 3/4*tau^(-5/2)*Y.eps_x - exp(mu)*((Y.d1'*epsi)*(3/4)*tau^(-5/2) + ...
                                              (Y.d1'*(epsi.^2))*(tau^-3)/4) - ...
            (2*X.P.taua - 1)/(2*tau^2);
  H(1, 2) = exp(mu)*(Y.d1'*epsi)/(2*tau^(3/2)) - ...
            X.P.mutau*(mu - X.P.mumu);
  H(2, 1) = H(1, 2);
end

%Newton-Raphson iterations for mu,tau
function [mutau, H, Hinv, is_max, use_LS] = mutauNR(mu, tau, epsi, X, Y)
  is_max = 0; %if we converged to a maximum
  is_NR_bad = 0; %if regular N-R iterations are insufficent,
                 %so we need to employ line search
  use_LS = 0; %whether we used linesearch (for display purposes only)

  mu_prev = mu;
  tau_prev = tau;

  i = 1;
  while true,
    tau_e = exp(tau);

    Y.exp_eps_tau = exp(epsi/sqrt(tau_e));
    Y.d1 = Y.exp_eps_tau;

    grad = gradmutau(mu, tau_e, , epsi, X, Y)';
    H = hessmutau(mu, tau_e, , epsi, X, Y);

    %change-of-variable chain rule factors
    H(2, 2) = grad(2)*tau_e + tau_e^2*H(2, 2);
    H(1, 2) = tau_e*H(1, 2); H(2, 1) = H(1, 2);
    grad(2) = grad(2)*tau_e;

    %change-of-variable Jacobian factors
    grad(2) = grad(2) + 1;

    %if Hessian is problematic, rewind an iteration and use line search by default
    if rcond(H) < eps || any(isnan(H(:))) || any(isinf(H(:))),
      %if we're in a problematic region at the first iteration, break and hope for the best
      if i == 1,
	H = eye(2); Hinv = -eye(2);
	disp('WARNING: Full conditional shifted significantly since last iteration!');

	break
      end

      is_NR_bad = 1;
      i = i - 1;

      mu = mu_prev;
      tau = tau_prev;

      continue
    else
      is_NR_bad = 0;
    end

    %check if Hessian is negative definite
    is_ndef = ~is_NR_bad && all(eig(H) < 0);

    %1. if Newton-Raphson will work fine, use it
    Hinv = inv(H);
    step = -Hinv*grad;

    %check if we've reached a local maximum
    if norm(grad) <= 1e-6 && is_ndef, is_max = 1; break; end

    %2. otherwise, employ line search
    fc_step = fcmutau(mu + step(1), tau + step(2), , epsi, , , X, Y);
    if is_NR_bad || isnan(fc_step) || isinf(fc_step) || ...
       fc_step - fcmutau(mu, tau, epsi, X, Y) < -1e-3,
      %indicate that we used linesearch for these iterations
      use_LS = 1;

      %2.1. ensure N-R direction is even ascending.  if not, use direction of gradient
      if ~is_ndef, step = grad; end

      %2.2. regardless of the method, perform line search along direction of step
      s0 = norm(step);
      d_hat = step/s0;

      %bound line search from below at current value
      fc = fcmutau(mu, tau, , epsi, , , X, Y)*[1 1];

      %2.3. do line search
      for l = 0:50,
	s = s0*0.5^l;

	f = fcmutau(mu + s*d_hat(1), tau + s*d_hat(2), epsi, X, Y);
	if fc(mod(l - 1, 2) + 1) > fc(mod(l - 2, 2) + 1) && fc(mod(l - 1, 2) + 1) > f,
	  s = s0*0.5^(l - 1);
	  break;
	else
	  fc(mod(l, 2) + 1) = f;
	end
      end

      step = d_hat*s;
    end

    %update mu,tau
    mu_prev = mu; tau_prev = tau;
    mu = mu + step(1); tau = tau + step(2);

    i = i + 1;
    if i > X.P.tMH_NR_iter,
      if ~is_ndef,
	H = eye(2); Hinv = -eye(2);
	disp('WARNING: Newton-Raphson terminated at non-concave point!');
      end

      break;
    end
  end
  mutau = [mu; tau];
end




%Epsilon M-H functions
%%%%%%%%%%%

X.x = [9.9773; 5.1288; 10.6584; 2.5370; 1.7598; 12.2921];
X.P.epsi_nu = 5;
X.len = length(X.x)

mu = 3;
epsi = [1.42; 11.52; 4.27; 7.79; 16.3; 1.89];
tau = 1;


%a M-H sample from p(epsi|-)
function [epsi, mrej] = epsisamp(epsi, tau, mu, X),
  %assumes no covariance between epsilons; does not sample as a single block

  %Newton-Raphson iterations to find proposal density
  [muF, HF, HFinv] = epsiNR(epsi, mu, tau, X);

  %now propose with multivariate t centered at epsiMLE with covariance matrix from Hessian
  %note that since Hessian is diagonal, we can just simulate from n univariate t's.
  epsiP = muF + sqrt(-HFinv).*trnd(X.P.epsi_nu, X.len, 1);
  %epsiP = normrnd(muF, -HFinv);

  arat = pratepsi(epsi, epsiP, tau, mu, X) + ...
         tqrat(epsi, epsiP, muF, muF, sqrt(-HFinv), sqrt(-HFinv), X.P.epsi_nu);

  ridx = log(rand(X.len, 1)) >= min(0, arat);
  epsi(~ridx) = epsiP(~ridx);
  mrej = mean(~ridx);
end

function qrat = tqrat(th0, thP, muF, muR, sigF, sigR, nu)
  qrat = -log(sigR) - (nu + 1)/2*log(1 + (th0 - muR).^2./(nu*sigR.^2)) - ...
         (-log(sigF) - (nu + 1)/2*log(1 + (thP - muF).^2./(nu*sigF.^2)));
end

%log posterior ratios for epsilon vector
function pr = pratepsi(epsi, epsiP, tau, mu, X)
  pr = epsiP.*X.x./sqrt(tau) - exp(mu + epsiP./sqrt(tau)) - epsiP.^2/2 - ...
       (epsi.*X.x./sqrt(tau) - exp(mu + epsi./sqrt(tau)) - epsi.^2/2);
end

%gradient of log p(epsilon|-)
function gr = gradepsi(epsi, tau, mu, X)
  gr = X.x./sqrt(tau) - exp(mu + epsi./sqrt(tau))./sqrt(tau) - epsi;
end

%Hessian of log p(epsilon|-)
function [H, Hinv] = hessepsi(epsi, tau, mu, X)
  %because Hessian is diagonal, we store as a column vector
  H = -exp(mu + epsi./sqrt(tau))./tau - 1;
  Hinv = 1./H;
end

%Newton-Raphson iteration for forward and reverse jumps
function [epsi, H, Hinv] = epsiNR(epsi, mu, tau, X),
  for i = 1:100,
    [H, Hinv] = hessepsi(epsi, tau, mu, X);

    %N-R update
    grad = gradepsi(epsi, tau, mu, X);
    epsi = epsi - Hinv.*grad;

    %we've reached a local maximum
    if norm(grad) < 1e-6, break; end
  end
end

[mu, tau, rej, use_LS] = mutausamp(epsi, tau, mu, X);

fprintf(["sampled mu: %s"], mu)
fprintf(["sampled tau: %s"], tau)
