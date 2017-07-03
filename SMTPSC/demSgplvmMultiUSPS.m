% Demo program for 3 classes

clc;clear all; close all; st = fclose('all');

randn('seed', 1e5);
rand('seed', 1e5);
Threshold = 1e-1;
jitter = 0.01;				% stabilization of covariance

dataSetName = 'MultiUSPS';
experimentNo = 1;
dataLabels = [0,1,2,3,4];
nTrOfEachClass = 20;
nTeOfEachClass = 100;

type = 'smcgplvm';
nKnn = 10;

latentDim = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


GPCPATH  = ['']; % this is where the GP code sits
DATAPATH = ['']; % this is the data directory
outfile = [];
path(GPCPATH,path)			% set up the path for GP routines
col = [];

% load data
[ x_tr,out_tr,x_te,out_te ] = loadMultiUSPS( dataLabels, nTrOfEachClass, nTeOfEachClass );

%-----------         Set up model x = f(z)       -------------%



options = fgplvmOptions('ftc');%fitc
options.optimiser = 'optimiMinimize';%scg
options.gType = type;
d = size(x_tr, 2);

modelR = fgplvmCreate(latentDim, d, x_tr, options);
params_R = fgplvmExtractParam(modelR);
startVal = 1;
endVal = modelR.N*modelR.q;
z = reshape(params_R(startVal:endVal), modelR.N, modelR.q); %z0
startVal = endVal+1;
endVal = endVal +modelR.kern.nParams;
modelR.kern.hyper = reshape(params_R(startVal:endVal), modelR.kern.nParams,1);


%------------        Set up model y = g(z)      --------------%

meth = 'ml';                % use MAP estimate only
% meth = 'ml_hmc';			% use MAP as inital w for HMC
% other options are meth = 'ml' or meth = 'hmc'

npc = latentDim + 2;		% number of parameters per class
rand('state',0);				% set the seed
randn('state',0);

m = size(out_tr,2);					% number of classes
hyper = rand(m*npc, 1);			% initial paramters

parvec = pak(m, size(x_tr,1), size(x_te,1)); % a vector of useful parameters

% MAP hyperparameter SCG search
options = zeros(1,18);		% Default options vector.
options(1) = 1;			% Display error values
options(14) = 25;		% Number of iterations
options(9) = 0;			% 1 => do a gradient check
 
% HMC options
hmcopt(1) = 10;			% number of retained samples
hmcopt(2) = 10;			% trajectory length
hmcopt(3) = 5;			% burn in
hmcopt(4) = 0.2;		% step size


% Set up the Gaussian hyperprior distributions for the parameters.
% For M independent classes, there are M different sets
% of covariance	parameters to specify.
% For each class, the first component is the scale
% and the last is the bias. 

% scale and bias:
for ci = 1:m
  mean_prior(ci,1) = -3;		% mean scale
  var_prior(ci,1) = 9;			% variance of scale
  mean_prior(ci,npc) = -3;		% mean bias
  var_prior(ci,npc) = 9;		% variance of bias
end

% input attribute hyperparameters:
for ci = 1:m
  mean_prior(ci,2:npc-1)  = -3.*ones(1,npc-2); 
  var_prior(ci,2:npc-1)   = 9.*ones(1,npc-2);
end


d = size(x_tr, 2);
z = ppcaEmbed(x_tr, latentDim); 

global model
model.modelR = modelR;
model.gType = type;
model.X = x_tr;
model.Y = out_tr;
model.Z = z;
model.XX = x_te;
model.YY = out_te;
[model.N, model.D] = size(x_tr);
model.M = m;        % Number of Class
model.p = latentDim;
model.isMissingData = 0;
model.isTranspose = 0;

kern.hyper = hyper;
kern.type = 'covMulitRbfArd';
kern.length = size(hyper,1);
model.kern = kern;
model.approx = 'cumGauss';
model.optimiser = 'minimize'; % scg or minimize


%---------------      Training         ---------------%
model.trainModel = 'combined';
oldZ = model.Z(:);
for i = 1:30
    hyper_w = [model.kern.hyper; model.Z(:)];
    newHyper_X = driverm(dataSetName,col,outfile,DATAPATH,x_tr,out_tr,x_te,out_te,meth,...
            options,hyper_w,hmcopt,mean_prior,var_prior,jitter);

    [ model.kern.hyper, model.Z ] = parseParam( newHyper_X, model );
    % model.kern.hyper = threshold(vec2mitheta(hyper,m),100);
    
    % Optimise the model.
    iters = -20; %1000
    display = 1;
    model.modelR.X = model.Z;
    % model.modelR = fgplvmExpandParam(model.modelR, hyper_z);
    model.modelR = fgplvmOptimise(model.modelR, display, iters);
    model.Z = model.modelR.X;
        
    diffZ = model.Z(:) - oldZ;
    sumDiffZ = sum(diffZ.*diffZ)
    if sumDiffZ < Threshold
        break;
    else
        oldZ = model.Z(:);
    end
end

%---------------      Testing         ---------------%

zz = zeros(size(model.XX, 1), latentDim);
for i = 1:size(model.XX, 1)
  llVec = fgplvmPointLogLikelihood(model.modelR, model.modelR.X, repmat(model.XX(i, :), model.modelR.N, 1));
  [void, ind] = max(llVec);
  zz(i, :) = model.modelR.X(ind, :);
end
% zz = model.modelR.X;

% Optimise X
% zz = ppcaEmbed(xx, latentDim);
% zc = zeros(size(zz, 1), latentDim);
zc = zeros(size(zz, 1), latentDim);
for i =1:size(zz, 1)
  zc(i, :) = fgplvmOptimisePoint(model.modelR, zz(i, :), model.XX(i, :), 0, -100);
end
 
y = smgpTransformLabel( model.Y );
yy = smgpTransformLabel( model.YY );
zplusY = [model.Z y];

[resultClass, classes, distance, voteMatrix] = kNN(zplusY, zc, nKnn, model);

nTest = length(yy);
% result = zeros(nTest, model.M);
% for ci = 1:model.M
%     result = resultClass(:,ci) - yy;
%     res = tabulate(result)
% end
result = zeros(nTest,1);
for i = 1:nTest
    ret = zeros(model.M, model.M);
    for ci = 1:model.M
        ret(:,ci) = voteMatrix(:,i,ci);
    end
    [result(i),ct] = find(ret == max(max(ret)), 1, 'first');
end
res = tabulate(result-yy)

filename = ['demSgplvm' dataSetName 'Asyn' 'Tr' num2str(size(model.Z, 1)) 'L' num2str(model.p)];
plotZ(model.Z, y, filename,1);
filename = ['demSgplvm' dataSetName 'Asyn' 'Te' num2str(size(zz, 1)) 'L' num2str(model.p)];
plotZ(zc, yy, filename,1);

save([filename]);





