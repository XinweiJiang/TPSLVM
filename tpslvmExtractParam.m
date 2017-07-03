function params = tpslvmExtractParam( model )
%TPSLVMEXTRACTPARAM Summary of this function goes here
%   Detailed explanation goes here

if model.isMAP
    params = [model.beta; model.gamma];
else
    params = [model.beta];
end

if isfield(model, 'back') && ~isempty(model.back)
    backParams = modelExtractParam(model.back);
    params = [backParams'; params ];
else
    params = [model.X(:); params];
end

if isfield(model, 'dynamics') && ~isempty(model.dynamics)
    dynParams = modelExtractParam(model.dynamics);
    params = [params; dynParams'];
end

end

