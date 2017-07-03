clc;clear all; close all; st = fclose('all');
dataSetName = 'OrlFT';
Opt.itersTrain = -200;
Opt.itersTest = -100;
Opt.nKnn = 5;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?

% Load data
load('ORL_64x64.mat');
x = fea/255;
y = gnd;
xx = x(1:5,:);
yy = y(1:5,:);

% Set up and optimise model
for latentDim = [100:-5:10]
% for latentDim = [10,15,20,25,30,35,40,45,50]
    [z] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end

