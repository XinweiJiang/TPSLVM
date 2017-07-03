function err = lvmNearestNeighbour(model, lbls)

% LVMNEARESTNEIGHBOUR Give the number of errors in latent space for 1 nearest neighbour.
%
%	Description:
%
%	LVMNEARESTNEIGHBOUR(MODEL, LBLS) computes the number errors for 1
%	nearest neighbour in latent space.
%	 Arguments:
%	  MODEL - the model for which the computation is required.
%	  LBLS - the labels of the data.


%	Copyright (c) 2004, 2006, 2008 Neil D. Lawrence
% 	lvmNearestNeighbour.m SVN version 24
% 	last update 2008-06-13T12:53:24.000000Z

X = model.X';
d = dist2(X, X);
for i = 1:size(X, 1); 
  d(i, i) = inf; 
end

if size(lbls,2) > 1
        lbls = smgpTransformLabel( lbls );
end
    
[void, ind] = min(d);
lblsTe = lbls(ind);
err = size(X, 1) - sum(lblsTe == lbls);
