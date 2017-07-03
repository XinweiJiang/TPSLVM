clc;clear all; close all; st = fclose('all');
randn('seed', 1e5);
rand('seed', 1e5);
Threshold = 1e-1;

Opt.itersTrain = -100;
Opt.itersTest = -100;
Opt.approx = 'ftc';
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 0;          % use MAP/ML?
Opt.isOptGamma = 0;     % optimize gamma(only when used MAP)?
dataSetName = 'WineDC';
% dataLabels = [0,1,2,3,4];
nTrOfEachClass = 30;
nTeOfEachClass = 1000;
filename = ['retTpslvm' dataSetName 'Accuracy' ];
Acc = [];

% load data
[ x,y ] = loadData( 'wine.data' );
[ x, y ] = smgpSort( x, y );
[ x, y, xx, yy ] = sgpDivTrainTestData( x, y, nTrOfEachClass, nTeOfEachClass );
% y = smgpTransformLabel( y );
% yy = smgpTransformLabel( yy );

for latentDim = [1,2,3,4,5,7,9,10,12]
% for latentDim = [7]
    [ z, zz, retAcc ] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
    Acc = [Acc; [latentDim size(x,1) size(xx,1) retAcc]];
end

save(filename, 'Acc');
