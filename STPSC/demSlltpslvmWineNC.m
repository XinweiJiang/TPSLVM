clc;clear all; close all; st = fclose('all');

Opt.iters = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
dataSetName = 'WineNC';

filename = ['retSlltpslvm' dataSetName 'Accuracy' ];
Acc = [];

for i = [10:5:40]
    nTrOfEachClass = i;
    nTeOfEachClass = 1000;

    % load data
    [ x,y ] = loadData( 'wine.data' );
    [ x, y ] = smgpSort( x, y );
    [ x, y, xx, yy ] = sgpDivTrainTestData( x, y, nTrOfEachClass, nTeOfEachClass );
    if (size(y,2) == 1)
        y = smgpTransformLabel( y );
        yy = smgpTransformLabel( yy );
    end
    if size(y,2) == 1
        nClass = length(unique(y));
    else
        nClass = size(y,2);
    end

    % for latentDim = [1,2,3,4,5,7,9,10,12]
    for latentDim = [5,10]
        if nClass > 2
            [z, zz, retAcc]  = MultiSlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
        else
            [z, zz, retAcc]  = BinarySlltpslvm( dataSetName, x, y, xx, yy, latentDim, Opt );
        end
        Acc = [Acc; [latentDim size(x,1) size(xx,1) retAcc]];
    end
end
% Acc = sortrows(Acc);
save(filename, 'Acc');