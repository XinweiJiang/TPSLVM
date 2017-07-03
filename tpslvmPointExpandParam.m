function model = tpslvmPointExpandParam(model, params)
%TPSLVMEXPANDPARAM Summary of this function goes here
%   Detailed explanation goes here

nLen = length(params);

model.XX = reshape(params(1:nLen), model.q, 1);

model.Xstar = [model.X model.XX];
model.Estar = tpsKernCompute( model.Xstar);
model.Tstar = [model.Xstar;ones(1,model.N+1)];

model.Kstar = 1/model.beta.*eye(size(model.Estar,1))+model.Estar'*model.Estar+model.Tstar'*model.Tstar;
[model.invKstar, U] = pdinv(model.Kstar);
model.logDetKstar = logdet(model.Kstar, U);

end

