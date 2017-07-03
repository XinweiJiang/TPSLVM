function [k, sk] = whiteKernCompute(kern, x, x2)

% WHITEKERNCOMPUTE Compute the WHITE kernel given the parameters and X.
%
%	Description:
%
%	K = WHITEKERNCOMPUTE(KERN, X, X2) computes the kernel parameters for
%	the white noise kernel given inputs associated with rows and
%	columns.
%	 Returns:
%	  K - the kernel matrix computed at the given points.
%	 Arguments:
%	  KERN - the kernel structure for which the matrix is computed.
%	  X - the input matrix associated with the rows of the kernel.
%	  X2 - the inpute matrix associated with the columns of the kernel.
%
%	K = WHITEKERNCOMPUTE(KERN, X) computes the kernel matrix for the
%	white noise kernel given a design matrix of inputs.
%	 Returns:
%	  K - the kernel matrix computed at the given points.
%	 Arguments:
%	  KERN - the kernel structure for which the matrix is computed.
%	  X - input data matrix in the form of a design matrix.
%	
%
%	See also
%	WHITEKERNPARAMINIT, KERNCOMPUTE, KERNCREATE, WHITEKERNDIAGCOMPUTE


%	Copyright (c) 2004, 2005, 2006, 2009 Neil D. Lawrence
% 	whiteKernCompute.m CVS version 1.4
% 	whiteKernCompute.m SVN version 328
% 	last update 2009-04-23T11:33:43.000000Z

if nargin < 3
  sk = speye(size(x, 1));
  k = kern.variance*sk;
else
  k = spalloc(size(x, 1), size(x2, 1), 0);
end
