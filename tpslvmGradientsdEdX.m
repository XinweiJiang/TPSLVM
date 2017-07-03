function dEdXmn = tpslvmGradientsdEdX( X, m, n )
%TPSLVMGRADIENTSDEDX Summary of this function goes here
%   Detailed explanation goes here

[q,N] = size(X);
n1 = dist11(X, m, n);
n2 = dist21(X, X);


% dEdXmn = n1.*(log(n2)+ones(N,N))./(16*pi);

c2 = 1/(16*sqrt(pi));
c4 = -c2;


if q == 2
    dEdXmn = c2.*n1.*(log(n2)+ones(N,N));
elseif q == 4
    dEdXmn = c4.*n1;
else
    c0 = gamma(q/2-2)/(16*sqrt(pi^q))*(4-q)/2;
    dEdXmn = c0.*n1;
end

end

