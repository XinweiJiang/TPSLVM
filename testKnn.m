zz = modelOut(model.back, xx');
% zz = zz';
y = y';yy = yy';
    if size(y,2) > 1
        y = smgpTransformLabel( y );
        yy = smgpTransformLabel( yy );
    end
    zplusY = [z' y];
    [resultClass, classes, distance] = kNN_TPSLVM(zplusY, zz, 5, model.beta);
    res = tabulate(resultClass - yy)
    retAcc = res(find(res(:,1)==0),3);