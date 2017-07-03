function E = tpsKernCompute( x, xx )
%TPSKERNCOMPUTE Summary of this function goes here
%   Detailed explanation goes here

if nargin == 1
    xx = x;
end

q = size(x,1);
n2 = dist21(x, xx);
% E = n2.*log(n2)./(16*pi);

c2 = 1/(16*sqrt(pi));
c4 = -c2;

E = c2.*n2.*log(n2);

% if q == 2
%     E = c2.*n2.*log(n2);
% elseif q == 4
%     E = c4.*log(n2);
% else
%     c0 = gamma(q/2-2)/(16*sqrt(pi^q));
%     E = c0.*(sqrt(n2).^(4-q));
% end

end

