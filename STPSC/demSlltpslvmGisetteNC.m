clc;clear all; close all; st = fclose('all');

Opt.iters = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
dataSetName = 'GisetteNC';
filename = ['retSlltpslvm' dataSetName 'Accuracy' ];
Acc = [];

for i=[10:10:100]

    nTrOfEachClass = i;
    nTeOfEachClass = 1000;
    % load data
    load('Gisette_Scale.mat');
    [ x, y, xx0, yy0 ] = sgpDivTrainTestData( x, y, nTrOfEachClass, nTeOfEachClass );
    % y = smgpTransformLabel( y );
    % yy = smgpTransformLabel( yy );
    if size(y,2) == 1
        nClass = length(unique(y));
    else
        nClass = size(y,2);
    end

    % latentDim = 3;
    for latentDim = [1,2,3,4,5,7,9,10,13,15]
        if nClass > 2
            [z, zz, retAcc]  = MultiSlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
        else
            [z, zz, retAcc]  = BinarySlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
        end

        Acc = [Acc; [latentDim size(y,1) size(yy,1) retAcc]];
    end
    
    save(filename, 'Acc');
end
