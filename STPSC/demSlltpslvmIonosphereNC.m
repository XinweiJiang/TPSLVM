clc;clear all; close all; st = fclose('all');

Opt.iters = -200;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
dataSetName = 'IonosphereNC';
filename = ['retSlltpslvm' dataSetName 'Accuracy' ];
Acc = [];


for i = [10:10:100]
% for i = [80]
    
    nTrOfEachClass = i;%100
    nTeOfEachClass = 1000;%1000

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

%     for latentDim = [1,2,3,4,5,7,9,11,13,15]
%     for latentDim = [3,5,9,11,15]
    for latentDim = [11]
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
