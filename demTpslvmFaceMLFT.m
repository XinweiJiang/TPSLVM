clc;clear all; close all; st = fclose('all');
dataSetName = 'FaceFT';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 0;          % use MAP/ML?
Opt.isOptGamma = 0;     % optimize gamma(only when used MAP)?


% Load data
load('frey_rawface.mat');
ff = ff';
% ff = sgpNormalize( ff, 1 );
ff = double(ff)./255;
x = [];
nEachClass = 10;
for j=[1:43]%[1:43]
    x = [x; ff((j-1)*45+1 : (j-1)*45+nEachClass,  :)];
end
y = ones(size(x,1),1);
y(1:14*nEachClass) = 1;
y(14*nEachClass+1:29*nEachClass) = 2;
y(29*nEachClass+1:43*nEachClass) = 3;
%     [ x, y ] = smgpSort( x, y );
%     [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, 60, 500 );
xx = x(1:3,:);
yy = y(1:3,:);


% Set up and optimise model
for latentDim = [2]
    [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end
