function [ z, zz, retAcc ] = Tpslvm( dataSetName, x, y, xx, yy, latentDim, Opt )
%TPSLVM Summary of this function goes here
%   Detailed explanation goes here
% Parameters:
%     X : D x N;
%     XX : D x M;
%     Y : P x N;
%     YY : P x M;


type = 'tpslvm';
experimentNo = 1;

itersTrain = Opt.itersTrain;
itersTest = Opt.itersTest;
nKnn = Opt.nKnn;
isAutoClosePlot = Opt.isAutoClosePlot;
dataSetName = [upper(dataSetName(1)) dataSetName(2:length(dataSetName))];
fprintf('Dataset: %s; Latent Dimension:%d; Training Data: %d; Testing Data: %d\n', dataSetName,latentDim,size(x,2),size(xx,2));

if latentDim > size(x,1)
    fprintf('Latent Dimension (%d) is larger than training dimension (%d), return null directly.\n',latentDim, size(x,1));
    z = [];
    zz = [];
    retAcc = 0;
    return;
end

if latentDim > size(x,2)
    fprintf('Latent Dimension (%d) is larger than number of training samples (%d), return null directly.\n',latentDim, size(x,2));
    z = [];
    zz = [];
    retAcc = 0;
    return;
end

% Create the model.
options.optimiseInitBack = 0;
options.isMAP = Opt.isMAP;
if ~isfield(Opt, 'isBACK')
    options.isBACK = 0;
else
    options.isBACK = Opt.isBACK;
end
if ~isfield(Opt, 'isDYN')
    options.isDYN = 0;
else
    options.isDYN = Opt.isDYN;
end
if ~options.isDYN
    options.isMAP = 0;
end

switch options.isBACK
    case 1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        options.back = 'kbr';
        options.backOptions = kbrOptions(x');
        options.backOptions.kern = kernCreate(x', 'rbf');
        options.backOptions.kern.inverseWidth = 0.0001;
    case 2
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        options.back = 'mlp';
        options.backOptions = mlpOptions;
        options.optimiseInitBack = 0;
end
[D,N] = size(x); NN = size(xx,2);
model = tpslvmCreate(x, D, latentDim, options);

% Add dynamics model.
switch options.isDYN
    case 1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        optionsDyn = gpReversibleDynamicsOptions('ftc');
%         optionsDyn.kern.comp{1}.comp{1}.inverseWidth = 1;
        model = tpslvmAddDynamics(model, 'gpReversible', optionsDyn);
    case 2
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        optionsDyn = gpOptions('ftc');
        optionsDyn.kern = kernCreate(model.X', {'rbf', 'white'});
        optionsDyn.kern.comp{1}.inverseWidth = 0.01;
        % This gives signal to noise of 0.1:1e-3 or 100:1.
        optionsDyn.kern.comp{1}.variance = 1;
        optionsDyn.kern.comp{2}.variance = 1e-3^2;
        model = tpslvmAddDynamics(model, 'gpTime', optionsDyn);
    case 3
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        optionsDyn = gpOptions('ftc');
        optionsDyn.kern = kernCreate(model.X', {'rbf', 'white'});
        optionsDyn.kern.comp{1}.inverseWidth = 0.01;
        % This gives signal to noise of 0.1:1e-3 or 100:1.
        optionsDyn.kern.comp{1}.variance = 0.1^2;
        optionsDyn.kern.comp{2}.variance = 1e-3^2;
        model = tpslvmAddDynamics(model, 'gp', optionsDyn);
end

% Optimise the model.
display = 1;
model = tpslvmOptimise(model, display, itersTrain);
z = model.X;

zz = zeros(latentDim, NN);
if isfield(Opt, 'doTest') && Opt.doTest == 1
    % Test
    zc = transpose(ppcaEmbed(xx', latentDim));
    if ~ispc
        display = 0;
    end

    for i = 1:NN
        model = tpslvmPointCreate(xx(:,i), zc(:,i), model);
        model = tpslvmPointOptimise(model, display, itersTest);
        zz(:,i) = model.XX;
    end

    % Classify testing data with KNN
    y = y';yy = yy';
    if size(y,2) > 1
        y = smgpTransformLabel( y );
        yy = smgpTransformLabel( yy );
    end
    zplusY = [z' y];
    [resultClass, classes, distance] = kNN_TPSLVM(zplusY, zz', nKnn, model.beta);
    res = tabulate(resultClass - yy)
    retAcc = res(find(res(:,1)==0),3);
else
    retAcc = 0;
end

% Save the model.
strtemp = [];
if model.isMAP
    strtemp = [strtemp 'Map'];
end
if isfield(model, 'back') && ~isempty(model.back)
    strtemp = [strtemp 'Back' num2str(model.isBACK)];
end
if isfield(model, 'dynamics') && ~isempty(model.dynamics)
    strtemp = [strtemp 'Dyn' num2str(model.isDYN)];
end


filename = ['demTpslvm' dataSetName 'Tr' num2str(N) 'L' num2str(latentDim) strtemp];
plotZ(z', y', filename, isAutoClosePlot);


filename = ['demTpslvm' dataSetName 'Tr' num2str(N)  'Te' num2str(NN) 'L' num2str(latentDim) strtemp];
save(filename);

end

