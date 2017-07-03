clc;clear all; close all; st = fclose('all');
dataSetName = 'FaceFT';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?


% Load data
% load('frey_rawface_sameremoved.mat');
load('frey_rawface.mat');
x0 = double(ff')./255;

y0 = ones(size(x0,1),1);
y0(1:14*45) = 1;
y0(14*45+1:29*45) = 2;
y0(29*45+1:1965) = 3;

% Delete very close samples in the data set for numerical stability
[x, y] = delCloseSample( x0, y0, 1.5 );      

xx = x(1:3,:);
yy = y(1:3,:);


% Set up and optimise model
for latentDim = [2]
    [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end
