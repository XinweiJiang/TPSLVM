function g = tpslvmPointLogLikeGradients(model)

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

dXX = zeros(model.q, 1);
Ystar = model.Ystar;
Xstar = model.Xstar;
Tstar = model.Tstar;
Estar = model.Estar;
beta = model.beta;
gamma = model.gamma;
q = model.q;
Nplus = model.N+1;
D = model.D;


% pre-compute matrix
gKstar = localCovarianceGradientsStar(model);

% t = cputime;
% for m = 1:q
%         dKdXmn = tpslvmGradientsdKdX( Xstar, Estar, m, Nplus );
%         dXX(m,1) = trace(gKstar*dKdXmn);        
% end
% cputime - t

% dX0 = zeros(model.q, model.N);
for m = 1:q
    dKdXmn = tpslvmGradientsdKdX( Xstar, Estar, m, Nplus );
    for i = 1:Nplus
        dXX(m,1) = dXX(m,1)+gKstar(i,:)*dKdXmn(:,i);  %+ dLogP(X)/dXmn        
    end
end

g = [dXX(:)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V2.1@2011-11-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % pre-compute matrix
% invU_E = (beta*Estar*Estar'+eye(N))\Estar;
% dLogUdE = 2*beta*invU_E;
% 
% invEtE_Tt = (1/beta*eye(N)+Estar'*Estar)\Tstar';
% invF = (Tstar*invEtE_Tt+eye(q+1))\eye(q+1);
% 
% invK_Yt = (1/beta.*eye(N)+Estar'*Estar+Tstar'*Tstar)\Ystar';
% invK_Yt_Y_invK = invK_Yt*invK_Yt';
% 
% dLogFdT = 2*invF*invEtE_Tt';
% dLogFdE = 2*Estar*invEtE_Tt*invF*invEtE_Tt';
% dTrdE = 2*Estar*invK_Yt_Y_invK;
% dTrdT = 2*Tstar*invK_Yt_Y_invK;
% 
% % if model.isMAP == 1
% %     if model.isOptGamma == 1
% %         dLogPXdXmn = 2*gamma;
% %     else
% %         dLogPXdXmn =  2;
% %     end
% % else
%     dLogPXdXmn = 0;
% % end
% 
% for m = 1:q
%         dEdXmn = tpslvmGradientsdEdX( Xstar, m, N );
%         dXX(m,1) = -0.5*(D*trace(dLogUdE*dEdXmn)+...
%             D*dLogFdT(m,N)-...
%             D*trace(dLogFdE*dEdXmn)-...
%             trace(dTrdE*dEdXmn)-...
%             dTrdT(m,N)+...
%             dLogPXdXmn*Xstar(m,N));  %+ dLogP(X)/dXmn         
% end
% 
% g = [dXX(:)];



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


function gKstar = localCovarianceGradientsStar(model)

% LOCALCOVARIANCEGRADIENTS

invKy = model.invKstar*model.Ystar';
gKstar = 0.5*(-model.D*model.invKstar + invKy*invKy');

end    