clc;clear all; close all; st = fclose('all');
dataSetName = 'IrisFT';
Opt.itersTrain = -500;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 0;          % use MAP/ML?
Opt.isOptGamma = 0;     % optimize gamma(only when used MAP)?


% Load data
[x, y] = loadData('iris.modified.data');
x = sgpNormalize( x );
[ x, y ] = smgpSort( x, y );
xx = x(1:3,:);
yy = y(1:3,:);
%     [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, i, 500 );


% Set up and optimise model
latentDim = 2;
z = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
