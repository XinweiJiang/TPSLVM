function [ g ] = tpsKernCompute0( kern, x, z )
%TPSKERNCOMPUTE Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    g = covTps(kern.bias, x);
else
    [s, g] = covTps(kern.bias, x, z);
end

end

