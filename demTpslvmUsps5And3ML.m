clc;clear all; close all; st = fclose('all');
dataSetName = 'Usps5And3';
Opt.itersTrain = -10;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isOptBeta = 1;      % optimize beta?
Opt.isMAP = 0;          % use MAP/ML?
Opt.isOptGamma = 0;     % optimize gamma(only when used MAP)?

% for i = [10:10:200]
for i = [2]
    % Load data
    [x, y, xx, yy] = loadBinaryUSPS(3,5);
    [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, 100, 5 );

    % Set up and optimise model
    latentDim = i;
    z = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
end