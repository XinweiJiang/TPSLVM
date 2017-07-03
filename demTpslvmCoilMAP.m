clc;clear all; close all; st = fclose('all');
dataSetName = 'Coil';
Opt.itersTrain = -1000;
Opt.itersTest = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 0;
Opt.isMAP = 1;          % use MAP/ML?


for b = [0,1,2]
    for d = [0,1,2,3]
        Opt.isBACK = b;         % use back constrains?
        Opt.isDYN = d;          % use Dynamic?
        % Load data
        load('Coil20.mat');
        x = double(data{1})./255;
        y = ones(size(x,1),1);
        %     [ x, y ] = smgpSort( x, y );
        %     [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, 60, 500 );
        xx = x(1:3,:);
        yy = y(1:3,:);


        % Set up and optimise model
        for latentDim = [2]
            [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
        end
    end
end