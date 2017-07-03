function dKdXmn = tpslvmGradientsdKdX( X, E, m, n )
%TPSLVMGRADIENTSDEDX Summary of this function goes here
%   Detailed explanation goes here

d0 = tpslvmGradientsdEdX( X, m, n );
dEtEdXmn = d0*E+E*d0;

N = size(E,1);
d0 = zeros(N,N);
d1 = zeros(N,N);
d0(n,:) = X(m,:);
d1(:,n) = X(m,:)';
dTtTdXmn = d0+d1;

dKdXmn = dEtEdXmn+dTtTdXmn;

end

