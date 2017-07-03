function [ L0,L1,L2 ] = testTrace( Y, K )
%TESTTRACE Summary of this function goes here
%   Detailed explanation goes here

[n,m] = size(Y);
L0 = 0;

for i = 1:n
    L0 = L0+Y(i,:)*K*Y(i,:)';
end

L1 = trace(K*Y'*Y);
L2 = trace(Y*K*Y');


end

