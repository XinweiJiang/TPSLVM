function model = tpslvmExpandParam(model, params)
%TPSLVMEXPANDPARAM Summary of this function goes here
%   Detailed explanation goes here

startVal = 1;

if isfield(model, 'back') && ~isempty(model.back)
  % update modelParameters
  endVal = model.back.numParams;
  paramBack = params(startVal:endVal);
  model.back = modelExpandParam(model.back, paramBack');

  % update latent locations
  tmp = modelOut(model.back,model.Y');
  tmp_dim = 1;
  Xt = model.X';
  for(i = 1:1:model.q)
    if(length(find(model.back.indexOut==i))~=0)
      Xt(:,model.back.indexOut(tmp_dim)) = tmp(:,tmp_dim);
      tmp_dim = tmp_dim + 1;
    else
      startVal = endVal + 1;
      endVal = endVal + model.N;
      Xt(:,i) = reshape(params(startVal:endVal),model.N,1);
    end
  end
  clear tmp tmp_dim;
  model.X = Xt';
else
  endVal = model.q*model.N;
  model.X = reshape(params(startVal:endVal), model.q, model.N);
end

startVal = endVal+1;
endVal = startVal;
model.beta = params(startVal);

if model.isMAP
    startVal = endVal+1;
    endVal = startVal;
    model.gamma = params(startVal);
end

model.E = tpsKernCompute( model.X );
model.T = [model.X;ones(1,model.N)];

model.K = 1/model.beta.*eye(size(model.E,1))+model.E'*model.E+model.T'*model.T;

[model.invK, U] = pdinv(model.K);
model.logDetK = logdet(model.K, U);
% model.invK0 = model.K\eye(size(model.E,1));

% Give parameters to dynamics if they are there.
if isfield(model, 'dynamics') & ~isempty(model.dynamics)
  startVal = endVal + 1;
  endVal = length(params);

  % Fill the dynamics model with current latent values.
  model.dynamics = modelSetLatentValues(model.dynamics, model.X');

  % Update the dynamics model with parameters (thereby forcing recompute).
  paramDynamic = params(startVal:endVal);
  model.dynamics = modelExpandParam(model.dynamics, paramDynamic');
end

end

