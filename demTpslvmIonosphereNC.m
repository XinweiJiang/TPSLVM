clc;clear all; close all; st = fclose('all');

Opt.itersTrain = -100;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 0;          % use MAP/ML?
Opt.isOptGamma = 0;     % optimize gamma(only when used MAP)?

dataSetName = 'IonosphereNC';
filename = ['retTpslvm' dataSetName 'Accuracy' ];
Acc = [];

for i = [10:10:100]
% for i = [10]
    
    nTrOfEachClass = i;
    nTeOfEachClass = 1000;

    % load data
    [ x,y ] = loadData( 'ionosphere.modified.data' );
    [ x, y ] = smgpSort( x, y );
    [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, nTrOfEachClass, nTeOfEachClass );
    % y = smgpTransformLabel( y );
    % yy = smgpTransformLabel( yy );
    if size(y,2) == 1
        nClass = length(unique(y));
    else
        nClass = size(y,2);
    end

    % latentDim = 3;
    for latentDim = [3,5,9,11,15]
        [z, zz, retAcc] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
        Acc = [Acc; [latentDim size(y,1) size(yy,1) retAcc]];
    end

end

save(filename, 'Acc');
