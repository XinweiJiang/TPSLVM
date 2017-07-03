clc;clear all; close all; st = fclose('all');
dataSetName = 'ORL';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?

nTr = 2;    %number of training data in each class

% Load data
load('ORL_64x64.mat');
fea = fea/255;
x = fea;
y = gnd;
xx = fea;
yy = gnd;
x = sgpNormalize( x );

% Set up and optimise model
for latentDim = [2,5,10,15,20,25,30,35,40,45,50]
    [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end

% save(['Tpslvm' dataSetName 'forClassify'], 'x','y','xx','yy','z','zz');
