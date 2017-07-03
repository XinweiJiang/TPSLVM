function model = tpslvmCreate(y, D, latentDim, options)

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

if size(y,1)~= D
    error(['dimension of y is wrong.']);
end

[model.D, model.N] = size(y);
model.q = latentDim;
model.Y = y;
model.YtY = y'*y;

% Initilize X with ppca
% for ppcaEmbed function, input dimension has to be NxD while the dimension
% of y is DxN
model.X = transpose(ppcaEmbed(y', latentDim));

model.isMAP = options.isMAP;
model.isBACK = options.isBACK;
model.isDYN = options.isDYN;

model.optimiser = 'optimiMinimize';

if isfield(options, 'back') && ~isempty(options.back)
  if isstruct(options.back)
    model.back = options.back;
  else
    if ~isempty(options.back)
      model.back = modelCreate(options.back, model.D, model.q,options.backOptions);
      if(isfield(options.backOptions,'indexOut')&&~isempty(options.backOptions.indexOut))
        model.back.indexOut = options.backOptions.indexOut;
      else
        model.back.indexOut = 1:1:model.q;
      end
    end
  end
  Xt = model.X';
  if options.optimiseInitBack
    % Match back model to initialisation.
    model.back = mappingOptimise(model.back, model.Y', Xt(:,model.back.indexOut));
  end
  % Now update latent positions with the back constraints output.
  Xt(:,model.back.indexOut) = modelOut(model.back, model.Y');
  model.X = Xt';
else
  model.back = [];
end

model.E = tpsKernCompute( model.X );
model.T = [model.X;ones(1,model.N)];
model.beta = 1;
model.gamma = 1;

model.constraints = {};

model.dynamics = [];

initParams = tpslvmExtractParam(model);
model.numParams = length(initParams);
% This forces kernel computation.
model = tpslvmExpandParam(model, initParams);
 
end

