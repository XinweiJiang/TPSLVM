function [g,dp] = tpslvmLogLikeGradients(model, gK)

% TPSLVMLOGLIKEGRADIENTS Compute the gradients for the FGPLVM.
%
%	Description:
%
%	G = FGPLVMLOGLIKEGRADIENTS(MODEL) returns the gradients of the log
%	likelihood with respect to the parameters of the GP-LVM model and
%	with respect to the latent positions of the GP-LVM model.
%	 Returns:
%	  G - the gradients of the latent positions (or the back
%	   constraint's parameters) and the parameters of the GP-LVM model.
%	 Arguments:
%	  MODEL - the FGPLVM structure containing the parameters and the
%	   latent positions.
%
%	[GX, GPARAM] = FGPLVMLOGLIKEGRADIENTS(MODEL) returns the gradients
%	of the log likelihood with respect to the parameters of the GP-LVM
%	model and with respect to the latent positions of the GP-LVM model
%	in seperate matrices.
%	 Returns:
%	  GX - the gradients of the latent positions (or the back
%	   constraint's parameters).
%	  GPARAM - gradients of the parameters of the GP-LVM model.
%	 Arguments:
%	  MODEL - the FGPLVM structure containing the parameters and the
%	   latent positions.
%	
%	
%
%	See also
%	FGPLVMLOGLIKELIHOOD, FGPLVMCREATE, MODELLOGLIKEGRADIENTS


%	Copyright (c) 2005, 2006, 2009 Neil D. Lawrence


%	With modifications by Carl Henrik Ek 2009
% 	fgplvmLogLikeGradients.m CVS version 1.6
% 	fgplvmLogLikeGradients.m SVN version 536
% 	last update 2009-09-28T08:45:30.000000Z

dX = zeros(model.q, model.N);
E = model.E;
T = model.T;
Y = model.Y;
X = model.X;
beta = model.beta;
gamma = model.gamma;
q = model.q;
N = model.N;
D = model.D;
invK = model.invK;

% pre-compute matrix
if nargin < 2
    gK = localCovarianceGradients(model);
end

% if model.isMAP
%     dLogPXdX = -gamma*X;
% else
%     dLogPXdX = zeros(size(X));
% end

% for m = 1:q
%     for n = 1:N
%         dKdXmn = tpslvmGradientsdKdX( X, E, m, n );
%         dX(m,n) = trace(gK*dKdXmn)+0.5*dLogPXdX(m,n);  %+ dLogP(X)/dXmn         
%     end
% end

% for m = 1:q
%     for n = 1:N
%         dKdXmn = tpslvmGradientsdKdX( X, E, m, n );
%         for i = 1:N
%             dX(m,n) = dX(m,n)+gK(i,:)*dKdXmn(:,i)+0.5*dLogPXdX(m,n);  %+ dLogP(X)/dXmn        
%         end
%     end
% end

for m = 1:q
    for n = 1:N
        dKdXmn = tpslvmGradientsdKdX( X, E, m, n );
        dX(m,n) = gK(:)'*dKdXmn(:);%+dLogPXdX(m,n);  %+ dLogP(X)/dXmn        
    end
end

dBeta = trace(-1/beta/beta.*gK);
% dBeta = -0.5*(-D*trace(1/beta/beta.*invK)+...
%         trace(1/beta/beta.*Y*invK*invK*Y'));

dParam = [dBeta];
% Check if Dynamics kernel is being used.
if isfield(model, 'dynamics') && ~isempty(model.dynamics)
    % Get the dynamics parameters
    dDynParam = modelLogLikeGradients(model.dynamics);
    % Include the dynamics latent gradients.
    dXdyn = modelLatentGradients(model.dynamics);
    dX = dX + dXdyn';
    
    dParam = [dParam;dDynParam'];
elseif isfield(model, 'isMAP') &&  model.isMAP
    dLogPXdX = -gamma*X;
    dGamma = -0.5*(-N*q/gamma+sum(sum(X.^2)));
    
    dX = dX + dLogPXdX; 
    dParam = [dParam;dGamma];
end

dX_or_back = tpslvmBackConstraintGrad(model, dX');

if nargin > 1
    dX_or_back = dX_or_back';
end


if nargout > 1
    g = dX_or_back;
    dp = dParam;
else
    g = [dX_or_back(:); dParam];
end
    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V2.1@2011-11-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % pre-compute matrix
% K = 1/beta.*eye(N)+E'*E+T'*T;
% invK = K\eye(N);
% dLogKdE = 2*invK*E;
% dLogKdT = 2*T*invK;
% 
% invK_Yt_Y_invK = invK*Y'*Y*invK;
% dTrKdE = 2*invK_Yt_Y_invK*E;
% dTrKdT = 2*T*invK_Yt_Y_invK;
% 
% if model.isMAP == 1
%     if model.isOptGamma == 1
%         dLogPXdXmn = 2*gamma;
%     else
%         dLogPXdXmn =  2;
%     end
% else
%     dLogPXdXmn = 0;
% end
% 
% for m = 1:q
%     for n = 1:N
%         dEdXmn = tpslvmGradientsdEdX( X, m, n );
%         dX(m,n) = -0.5*(D*trace(dLogKdE*dEdXmn)+...
%             D*dLogKdT(m,n)-...
%             trace(dTrKdE*dEdXmn)-...
%             dTrKdT(m,n)+...
%             dLogPXdXmn*X(m,n));  %+ dLogP(X)/dXmn         
%     end
% end
% 
% dBeta = -0.5*(-D*trace(1/beta/beta.*invK)+...
%         trace(1/beta/beta.*Y*invK*invK*Y'));
%     
% if model.isOptGamma == 1
%     dGamma = -0.5*(-N*q/gamma+sum(sum(X.^2)));
%     g = [dX(:);dBeta;dGamma];
% else
%     g = [dX(:);dBeta];
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V2.1@2011-11-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % pre-compute matrix
% invU_E = (beta*E*E'+eye(N))\E;
% dLogUdE = 2*beta*invU_E;
% 
% invEtE_Tt = (1/beta*eye(N)+E'*E)\T';
% invF = (T*invEtE_Tt+eye(q+1))\eye(q+1);
% 
% invK_Yt = (1/beta.*eye(N)+E'*E+T'*T)\Y';
% invK_Yt_Y_invK = invK_Yt*invK_Yt';
% 
% dLogFdT = 2*invF*invEtE_Tt';
% dLogFdE = 2*E*invEtE_Tt*invF*invEtE_Tt';
% dTrdE = 2*E*invK_Yt_Y_invK;
% dTrdT = 2*T*invK_Yt_Y_invK;
% 
% if model.isMAP == 1
%     if model.isOptGamma == 1
%         dLogPXdXmn = 2*gamma;
%     else
%         dLogPXdXmn =  2;
%     end
% else
%     dLogPXdXmn = 0;
% end
% 
% for m = 1:q
%     for n = 1:N
%         dEdXmn = tpslvmGradientsdEdX( X, m, n );
%         dX(m,n) = -0.5*(D*trace(dLogUdE*dEdXmn)+...
%             D*dLogFdT(m,n)-...
%             D*trace(dLogFdE*dEdXmn)-...
%             trace(dTrdE*dEdXmn)-...
%             dTrdT(m,n)+...
%             dLogPXdXmn*X(m,n));  %+ dLogP(X)/dXmn         
%     end
% end
% 
% dBeta = -0.5*(-N*D/beta+D*trace(invU_E*E')+...
%         D*trace(1/beta/beta.*invF*invEtE_Tt'*invEtE_Tt)+...
%         trace(1/beta/beta.*invK_Yt'*invK_Yt));
%     
% if model.isOptGamma == 1
%     dGamma = -0.5*(-N*q/gamma+sum(sum(X.^2)));
%     g = [dX(:);dBeta;dGamma];
% else
%     g = [dX(:);dBeta];
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V1.8@2011-11-07
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % pre-compute matrix
% dLogEdE0 = (beta*E*E'+eye(N))\E;
% dLogEdE = 2*beta*dLogEdE0;
% dLogTdT0 = (beta*T*T'+eye(q+1))\T;
% dLogTdT = 2*beta*dLogTdT0;
% 
% dTrEdE0 = (1/beta*eye(N)+E'*E)\Y';
% dTrEdE = dTrEdE0*dTrEdE0'*E;
% 
% dTrTdT0 = (1/beta*eye(N)+T'*T)\Y';
% dTrTdT = T*(dTrTdT0*dTrTdT0');
% 
% J = beta*(1/beta*eye(N)+E'*E)*(1/beta*eye(N)+T'*T);
% invJYt = J\Y';
% YinvJ = Y/J;
% dTrETdE0 = (1/beta*eye(N)+T'*T)*invJYt*YinvJ;
% dTrETdE = (dTrETdE0+dTrETdE0')*E';
% 
% dTrETdT0 = invJYt*YinvJ*(1/beta*eye(N)+E'*E);
% dTrETdT = T*(dTrETdT0+dTrETdT0');
% 
% % Compute derivatives
% for m = 1:q
%     for n = 1:N
%         dEdXmn = tpslvmGradientsdEdX( model.X, m, n );
%         dX(m,n) = -0.5*(D*trace(dLogEdE*dEdXmn)+...
%             D*dLogTdT(m,n)-...
%             8*trace(dTrEdE*dEdXmn)-...
%             10*dTrTdT(m,n)+...
%             4*beta*trace(dTrETdE*dEdXmn)+...
%             4*beta*dTrETdT(m,n));
%     end
% end
% 
% if model.isOptBeta == 1
%     dBeta = -0.5*(N*D/beta+D*trace(dLogEdE0*E')+...
%         D*trace(dLogTdT0*T')+...
%         trace(-4*Y*Y'+4/beta/beta*dTrEdE0'*dTrEdE0+...
%         5/beta/beta*dTrTdT0'*dTrTdT0-...
%         4*YinvJ*(1/beta/beta-E'*E*T'*T)*invJYt));
% 
%     g = [dX(:);dBeta];
% else
%     g = [dX(:)];
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end


function gK = localCovarianceGradients(model)

% LOCALCOVARIANCEGRADIENTS

invKy = model.invK*model.Y';
gK = 0.5*(-model.D*model.invK + invKy*invKy');

end    