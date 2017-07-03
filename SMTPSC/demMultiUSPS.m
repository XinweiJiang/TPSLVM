% Demo program for 3 classes

clc;clear all; close all; st = fclose('all');

randn('seed', 1e5);
rand('seed', 1e5);
Threshold = 1e-1;
jitter = 0.01;				% stabilization of covariance

type = 'smcgplvm';
dataSetName = 'MultiUSPS';
experimentNo = 1;
dataLabels = [0,1,2,3,4];
nTrOfEachClass = 20;
nTeOfEachClass = 100;
nKnn = 10;

GPCPATH  = ['']; % this is where the GP code sits
DATAPATH = ['']; % this is the data directory
path(GPCPATH,path)			% set up the path for GP routines
outfile = [];
col = [];

% load data
[ x_tr,out_tr,x_te,out_te ] = loadMultiUSPS( dataLabels, nTrOfEachClass, nTeOfEachClass );


latentDim = 2;


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
w = x_tr\z;

global model
% model.modelR = modelR;
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

hyper_w = [hyper; w(:)];

hyper_w = driverm(dataSetName,col,outfile,DATAPATH,x_tr,out_tr,x_te,out_te,meth,...
  options,hyper_w,hmcopt,mean_prior,var_prior,jitter);

[ model.kern.hyper, w ] = parseParam( hyper_w, model );

%---------------      Testing         ---------------%

model.Z = model.X*w;
y = smgpTransformLabel( model.Y );
yy = smgpTransformLabel( model.YY );
zplusY = [model.Z y];
zz = model.XX*w;
[resultClass, classes, distance, voteMatrix] = kNN(zplusY, zz, nKnn, model);

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

filename = ['demSllgplvm' dataSetName 'Tr' num2str(size(model.Z,1)) 'L' num2str(model.p)];
plotZ(model.Z,model.Y,filename,1);
filename = ['demSllgplvm' dataSetName 'Te' num2str(size(zz,1)) 'L' num2str(model.p)];
plotZ(zz,model.YY,filename,1);

save([filename '.mat']);






% 
% % Prediction options when using the hyperparameter sample(s)
% reject = 0;	% number of hyperparameter samples rejected when predicting
% gsmp = 100;	% number of activation samples in softmax posterior average
% 
% [m, ntr, nte, ntrte] = unpak(parvec);
% 
% [ty_all,tru_all]=max(out_trte);			% correct predictions
% tru = tru_all(1,ntr+1:ntrte);
% 
% % MAP: 
% 
% [meanpred_all] = final_pred([outfile,'.ml'], reject, gsmp, parvec);
% [py_all,pred_all]=max(meanpred_all');		% GP predictions
% pred = pred_all(1,ntr+1:ntrte);
% 
% 
% fprintf(1,'\n\n\nMAP Results\n')
% correct_pred = find(pred-tru==0);
% wrong_pred = find(pred-tru);
% fprintf(1,'test error rate = %f percent\n',100*length(wrong_pred)/nte)
% 
% % ARD
% fprintf(1,'\nMAP hyperparameters:')
% hyp_vec_all = getmat([outfile,'.ml.smp'],m*npc,0);
% hyp_mat_all = vec2mitheta(hyp_vec_all,m);
% fprintf(1,'\n    class1    class2    class3\n')
% hyp_mat_all(:,2:npc-1)'
% fprintf(1,'\n covariance scale:\n')
% hyp_mat_all(:,1)'
% fprintf(1,'\n covariance bias:\n')
% hyp_mat_all(:,npc)'
% 
% % HMC:
% 
% [meanpred_all] = final_pred([outfile,'.vhmc'], reject, gsmp, parvec);
% [py_all,pred_all]=max(meanpred_all');		% threshold the predictions
% pred = pred_all(1,ntr+1:ntrte);
% 
% fprintf(1,'\nHMC Results\n')
% correct_pred = find(pred-tru==0);
% wrong_pred = find(pred-tru);
% fprintf(1,'test error rate = %f percent\n',100*length(wrong_pred)/nte)
% 
% % ARD
% hyp_vec_all = getmat([outfile,'.vhmc.smp'],m*npc*hmcopt(1),0);
% hyp_mat_all = zeros(hmcopt(1),m,npc);
% for i = 1:hmcopt(1)
%  hyp_mat_all(i,:,:) = vec2mitheta(hyp_vec_all(1,1+(i-1)*m*npc:i*m*npc),m);
% end
% 
% fprintf(1,'\nmean hyperparameters:')
% fprintf(1,'\n    class1    class2    class3\n')
% temp = squeeze(mean(hyp_mat_all));
% temp(:,2:npc-1)'
% fprintf(1,'\n covariance scale:\n')
% temp(:,1)'
% fprintf(1,'\n covariance bias:\n')
% temp(:,npc)'
% 
% fprintf(1,'Standard deviation of the hyperparameters:')
% fprintf(1,'\n    class1    class2    class\n')
% temp2 = squeeze(std(hyp_mat_all));
% temp2(:,2:npc-1)'
% fprintf(1,'\n covariance scale:\n')
% temp2(:,1)'
% fprintf(1,'\n covariance bias:\n')
% temp2(:,npc)'





