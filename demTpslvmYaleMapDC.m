clc;clear all; close all; st = fclose('all');
dataSetName = 'Yale';
Opt.itersTrain = -200;
Opt.itersTest = -100;
Opt.nKnn = 5;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?

Acc = [];

for nTr = [5]    %number of training data in each class

% Load data
load('Yale_64x64.mat');
load(['Yale_64x64-' num2str(nTr) '-1.mat']);
x = fea(trainIdx,:)/255;
xx = fea(testIdx,:)/255;
y = gnd(trainIdx,:);
yy = gnd(testIdx,:);

% Set up and optimise model
for latentDim = [2:2:20]
% for latentDim = [10,15,20,25,30,35,40,45,50]
    [z, zz, retAcc] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
    Acc = [Acc; [latentDim size(y,1) size(yy,1) retAcc]];
end

end

filename = ['retTpslvm' dataSetName 'Accuracy' ];
save(filename, 'Acc');
