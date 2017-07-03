clc;clear all; close all; st = fclose('all');

Opt.iters = -100;
Opt.nKnn = 10;
Opt.isAutoClosePlot = 1;
dataSetName = 'Usps0To4NC';
dataLabels = [0,1,2,3,4];
filename = ['retSlltpslvm' dataSetName 'Accuracy' ];
Acc = [];


for i = [5]
    nTrOfEachClass = i;
    nTeOfEachClass = 500;

    % load data
    [ x,y,xx,yy ] = loadMultiUSPS( dataLabels, nTrOfEachClass, nTeOfEachClass );
    if size(y,2) == 1
        nClass = length(unique(y));
    else
        nClass = size(y,2);
    end
    
    DimArr = [5];
    % for latentDim = [1,2,3,4,5,7,9,10,13,15]
    for latentDim = DimArr
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
