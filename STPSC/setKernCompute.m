function g = setKernCompute( kern, x, x2 )
%SETKERNGRADX Summary of this function goes here
%   Detailed explanation goes here

%%%%%%'rbf', 'bias', 'white', 'tps'
switch kern.type
   case 'covTps'
       fhandle = str2func(['tpsKernCompute0']);
   case 'covSEiso'
       fhandle = str2func(['rbfKernCompute']);
    case 'covMulitRbfArd'
        fhandle = str2func(['rbfard2whiteKernCompute']);
    otherwise
        error('Unknown data set requested.')
end

if nargin < 3
  g = fhandle(kern, x);
else
  g = fhandle(kern, x, x2);
end

