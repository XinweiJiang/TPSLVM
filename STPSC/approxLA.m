function [alpha, sW, L, nlZ, dnlHpW] = approxLA(hyper_w, covfunc, lik, model)

% Laplace approximation to the posterior Gaussian Process.
% The function takes a specified covariance function (see covFunction.m) and
% likelihood function (see likelihoods.m), and is designed to be used with
% binaryGP.m. See also approximations.m.
%
% Copyright (c) 2006, 2007 Carl Edward Rasmussen and Hannes Nickisch 2007-03-29

persistent best_alpha best_nlZ        % copy of the best alpha and its obj value
tol = 1e-6;                   % tolerance for when to stop the Newton iterations
jitter = 0.001;

x = model.X;
y = model.Y;

startVal = 1;
endVal = model.kern.length; 
hyper = reshape(hyper_w(startVal:endVal), model.kern.length, 1);
model.kern.bias = hyper(1);  

switch model.gType
    case 'slltpslvm'
        startVal = endVal+1;
        endVal = endVal +  model.D*model.p;
        w = reshape(hyper_w(startVal:endVal), model.D, model.p);
        x = x*w;
        xt = x';
        model.modelR = tpslvmExpandParam(model.modelR, [xt(:); model.kern.bias]);
      
%      case 'gpgplvm'
%         startVal = endVal+1;
%         endVal = endVal +  model.N*model.p;
%         x = reshape(hyper_w(startVal:endVal), model.N, model.p);
%                 
%         if strcmp(model.trainModel, 'combined')
%             startVal = endVal+1;
%             endVal = endVal + model.modelR.kern.nParams;
%             model.modelR.y = x;
%             params_R = reshape(hyper_w(startVal:endVal), model.modelR.kern.nParams, 1);        
%             model.modelR = gpExpandParam(model.modelR, params_R');
%         end
                
    otherwise
        error('Unknown data set requested.')
end

[n, p] = size(x);

K = feval(covfunc{:}, hyper, x);                % evaluate the covariance matrix
% [K0, sk, n2] = rbfKernCompute(model.kern, x);
% a = K-K0;

if any(size(best_alpha) ~= [n,1])   % find a good starting point for alpha and f
  f = zeros(n,1); alpha = f;                                     % start at zero
  [lp,dlp,d2lp] = feval(lik,y,f,'deriv');   W=-d2lp;
  Psi_new = lp; best_nlZ = Inf; 
else
  alpha = best_alpha; f = K*alpha;                             % try best so far
  [lp,dlp,d2lp] = feval(lik,y,f,'deriv');   W=-d2lp;
  Psi_new = -alpha'*f/2 + lp;         
  if Psi_new < -n*log(2)                                 % if zero is better ..
    f = zeros(n,1); alpha = f;                                      % .. go back
    [lp,dlp,d2lp] = feval(lik,y,f,'deriv'); W=-d2lp; 
    Psi_new = -alpha'*f/2 + lp;
  end
end
Psi_old = -Inf;                                    % make sure while loop starts

while Psi_new - Psi_old > tol                        % begin Newton's iterations
  Psi_old = Psi_new; alpha_old = alpha; 
  sW = sqrt(W);                     
  L = chol(eye(n)+sW*sW'.*K);                            % L'*L=B=eye(n)+sW*K*sW
  b = W.*f+dlp;
  alpha = b - sW.*solve_chol(L,sW.*(K*b));
  f = K*alpha;
  [lp,dlp,d2lp,d3lp] = feval(lik,y,f,'deriv'); W=-d2lp;

  Psi_new = -alpha'*f/2 + lp;
  i = 0;
  while i < 10 && Psi_new < Psi_old               % if objective didn't increase
    alpha = (alpha_old+alpha)/2;                      % reduce step size by half
    f = K*alpha;
    [lp,dlp,d2lp,d3lp] = feval(lik,y,f,'deriv'); W=-d2lp;
    Psi_new = -alpha'*f/2 + lp;
    i = i+1;
  end 
end                                                    % end Newton's iterations

sW = sqrt(W);                                                    % recalculate L
L  = chol(eye(n)+sW*sW'.*K);                             % L'*L=B=eye(n)+sW*K*sW
nlZ = (alpha'*f/2 - lp + sum(log(diag(L))));      % approx neg log marg likelihood
    
if nargout >= 4                                        % do we want derivatives?
  dnlHP = zeros(model.kern.length,1);                    % allocate space for derivatives of the hyperparameters
  dnlK = zeros(n, n);                                   % allocate space for derivatives of the kernel
  %dnlZ = zeros(model.N, model.p);                                   % allocate space for derivatives of latent variables
  dnlW = zeros(model.D, model.p);                                   % allocate space for derivatives of W (Z=XW)
  dfdK = zeros(n, n);
  
  dnlK = (-alpha*alpha'+ L\(L'\diag(W)))/2;
  
  Z = repmat(sW,1,n).*solve_chol(L, diag(sW));
%   F0 = diag(W)\Z;                         %compute (I+KW)^{-1}
  F = 2.*Z - diag(diag(Z));
  C = L'\(repmat(sW,1,n).*K);
  s2 = 0.5*(diag(K)-sum(C.^2,1)').*d3lp;
  
  for j=1:length(hyper)
    dK = feval(covfunc{:}, hyper, x, j);
    s1 = alpha'*dK*alpha/2-sum(sum(Z.*dK))/2;
    b  = dK*dlp;
    s3 = b-K*(Z*b);
    dnlHP(j) = -s1-s2'*s3;
  end
  
  dnlK0 = zeros(n, n);

    for i = 1:n
      for j = 1:n
          b = smgpCreateFormatMatrix( i, j, dlp );
          s3 = b-K*(Z*b);
          dnlK0(i,j) = s2'*s3;
      end
    end

    dnlK = dnlK - dnlK0;
    gK = dnlK;
  
    switch model.gType
        case 'slltpslvm'
            [gZ, gHP] = tpslvmLogLikeGradients(model.modelR, gK);
            dnlW = sgpGradientsW( model, gZ );
            dnlHpW = [dnlHP;dnlW(:)];

%         case 'gpgplvm'
%             if strcmp(model.trainModel, 'combined')
%                 params_R = gpExtractParam(model.modelR);
%                 [f0, g0] = gpObjectiveGradient(params_R, model.modelR);
% 
%                 dnlHpW = [dnlHP;gZ(:);g0'];
%                 nlZ = nlZ + f0;
%             else
%                 dnlHpW = [dnlHP;gZ(:)];
%             end

        otherwise
            error('Unknown data set requested.')
    end
   
   
   if nlZ < best_nlZ                                            % if best so far ..
      best_alpha = alpha; best_nlZ = nlZ;           % .. then remember for next call
   end
 
end
