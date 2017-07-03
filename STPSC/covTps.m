function [A, B] = covTps(logtheta, x, z);

% TPS covariance function with a single hyperparameter. The covariance
% function is parameterized as:
%
% k(x^p,x^q) = x^p'*inv(P)*x^q + 1./t2;
%
% where the P matrix is t2 times the unit matrix. The second term plays the
% role of the bias. The hyperparameter is:
%
% logtheta = [ log(sqrt(t2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-27)

if nargin == 0, A = '1'; return; end              % report number of parameters

it2 = 1/logtheta;                                            % t2 inverse

if nargin == 2                                             % compute covariance
    N = size(x,1); 
    E = tpsKernCompute( x' );
    T = [x';ones(1,N)];
    
    A = it2*eye(N)+E'*E+T'*T;
elseif nargout == 2                              % compute test set covariances
    nX = size(x,1); nZ = size(z,1);
    x = [x;z];
    N = nX + nZ; 
    E = tpsKernCompute( x' );
    T = [x';ones(1,N)];
    
    AB = it2*eye(N)+E'*E+T'*T;
    A = diag(AB);
    A = A(nX+1:nX+nZ);
    B = AB(1:nX, nX+1:nX+nZ);
else                                                % compute derivative matrix
    N = size(x,1); 
    A = -it2*it2*eye(N);
end
