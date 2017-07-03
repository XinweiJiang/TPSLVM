clc;clear all; close all; st = fclose('all');
dataSetName = 'Vowels';
Opt.itersTrain = -500;
Opt.itersTest = -10;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
Opt.isMAP = 1;          % use MAP/ML?
% Opt.isBACK = 1;         % use back constrains?
% Opt.isDYN = 3;          % use Dynamic?

for b = [2]
    for d = [0]
        Opt.isBACK = b;         % use back constrains?
        Opt.isDYN = d;          % use Dynamic?
        
        % Load data
        load(dataSetName);
        [ x, y ] = smgpSort( x, y );
        [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, 50, 1 );
%         x = x(1:700,:);y = y(201:700,:);
        xx = x(1:4,:);yy = y(1:4,:);

        % Set up and optimise model
        for latentDim = [1:13]
            [z, zz] = Tpslvm( dataSetName, x', y', xx', yy', latentDim, Opt );
        end
    end
end