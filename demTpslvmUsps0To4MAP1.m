clc;clear all; close all; st = fclose('all');
randn('seed', 1e5);
rand('seed', 1e5);
Threshold = 1e-1;

Opt.itersTrain = -500;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 1;          % use MAP/ML?
Opt.isOptGamma = 1;     % optimize gamma(only when used MAP)?
Opt.isBACK = 2;
dataSetName = 'Usps0To4';
dataLabels = [0,1,2,3,4];
filename = ['retTpslvm' dataSetName 'Accuracy' ];
Acc = [];

for i = [50]
    nTrOfEachClass = i;
    nTeOfEachClass = 1000;

    % load data
    [ x,y,xx,yy ] = loadMultiUSPS( dataLabels, nTrOfEachClass, nTeOfEachClass );

    for latentDim = [1,2,3]
      [z, zz, retAcc] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
    end
end
