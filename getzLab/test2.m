Y.j = 1;
Y.d1 = 1

X.sums = sum(poissrnd(exp(-3. + 0.9 * randn(5000, 1)))


X.P.mumu = 

function gr = gradmutau(mu, tau, Beta, epsi, X, Y)
  gr = [X.sums(Y.j) - exp(mu)*sum(Y.d1) - ...
        X.P.mutau(Y.j)*tau*(mu - X.P.mumu(Y.j)), ...
        (-Y.eps_x + exp(mu)*(Y.d1'*epsi))/(2*tau^(3/2)) + ...
         (2*X.P.taua(Y.j) - 1)/(2*tau) - 1/X.P.taub(Y.j) - X.P.mutau(Y.j)/2*(mu - X.P.mumu(Y.j))^2];