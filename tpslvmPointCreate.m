function model = tpslvmPointCreate(yyi, xxi, model)

% TPSLVMCREATE Log-likelihood for a GP-LVM.
%
%	Description:
%
%	LL = FGPLVMLOGLIKELIHOOD(MODEL) returns the log likelihood for a
%	given GP-LVM model.
%	 Returns:
%	  LL - the log likelihood of the data given the model.
%	 Arguments:
%	  MODEL - the model for which the log likelihood is to be computed.
%	   The model contains the data for which the likelihood is being
%	   computed in the 'y' component of the structure.
%	
%	
%	
%
%	See also
%	GPLOGLIKELIHOOD, FGPLVMCREATE


%	Copyright (c) 2005, 2006, 2009 Neil D. Lawrence


%	With modifications by Carl Henrik Ek 2008, 2009
% 	fgplvmLogLikelihood.m CVS version 1.5
% 	fgplvmLogLikelihood.m SVN version 291
% 	last update 2009-03-04T22:08:40.000000Z

model.XX = xxi;
model.YY = yyi;

model.Ystar = [model.Y model.YY];
model.Xstar = [model.X model.XX];
model.Tstar = [model.Xstar;ones(1,model.N+1)];
model.Estar = tpsKernCompute( model.Xstar );
model.YstarTYstar = model.Ystar'*model.Ystar;
 
end

