function f = tpslvmPointLogLikelihood(model)

% TPSLVMLOGLIKELIHOOD Log-likelihood for a GP-LVM.
%
%	Description:
%
%	LL = FGPLVMLOGLIKELIHOOD(MODEL) returns the log likelihood for a
%	given GP-LVM model.
%	 Returns:
%	  LL - the log likelihood of the data given the model.
%	 Arguments:
%	  MODEL - the model for which the log likelihood is to be computed.
%	   The model contains the data for which the likelihood is being
%	   computed in the 'y' component of the structure.
%	
%	
%	
%
%	See also
%	GPLOGLIKELIHOOD, FGPLVMCREATE


%	Copyright (c) 2005, 2006, 2009 Neil D. Lawrence


%	With modifications by Carl Henrik Ek 2008, 2009
% 	fgplvmLogLikelihood.m CVS version 1.5
% 	fgplvmLogLikelihood.m SVN version 291
% 	last update 2009-03-04T22:08:40.000000Z

Ystar = model.Ystar;
Xstar = model.Xstar;
Tstar = model.Tstar;
Estar = model.Estar;

beta = model.beta;
gamma = model.gamma;
q = model.q;
N = model.N+1;
D = model.D;

f = -0.5*((D*model.logDetKstar)+...
    trace(model.invKstar*model.YstarTYstar));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V2.1@2011-11-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% invEtETt = (1/beta*eye(N)+Estar'*Estar)\Tstar';
% invEtETtTtY = (1/beta*eye(N)+Estar'*Estar+Tstar'*Tstar)\Ystar';
% 
% % if model.isMAP == 1
% %     if model.isOptGamma == 1
% %         logPX = -N*q*log(2*pi*gamma)+gamma*sum(sum(X.^2));
% %     else
% %         logPX = sum(sum(X.^2));
% %     end
% % else
%     logPX = 0;
% % end
% 
% f = -0.5*(D*log(det(beta*Estar*Estar'+eye(N)))+...
%     D*log(det(Tstar*invEtETt+eye(q+1)))+...
%     trace(invEtETtTtY*Ystar)+...
%     logPX);%+ LogP(X)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V1.8@2011-11-07
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% invEtE = (1/beta*eye(N)+E'*E)\eye(N);
% invTtT = (1/beta*eye(N)+T'*T)\eye(N);
% % J = beta*(1/beta*eye(N)+E'*E)*(1/beta*eye(N)+T'*T);
% % invJ = J\eye(N);
% invJ = 1/beta*invTtT*invEtE;
% 
% f = -0.5*(N*D*log(2*pi*beta)+...
%     D*log(det(beta*E*E'+eye(N)))+...
%     D*log(det(beta*T*T'+eye(q+1)))+...
%     trace(Y*(-4*beta*eye(N)+...
%     4*invEtE+...
%     5*invTtT-...
%     4*invJ)*Y'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
end

