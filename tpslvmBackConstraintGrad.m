function g = tpslvmBackConstraintGrad(model, gX)

% TPSLVMBACKCONSTRAINTGRAD Gradient with respect to back constraints if present.
%
%	Description:
%
%	TPSLVMBACKCONSTRAINTGRAD(MODEL, GX) converts the gradients of the
%	TPS-LVM model log likelihood with respect to the latent positions to
%	be gradients with respect to the parameters of the back constraints.
%	 Arguments:
%	  MODEL - the TPS-LVM model structure for which the conversion is to
%	   be done.
%	  GX - the gradients of the log likelihood with respect to the back
%	   constraint parameters.
%	


% Check for back constraints.
if isfield(model, 'back') & ~isempty(model.back)
  g_w = modelOutputGrad(model.back, model.Y');
  g_modelParams = zeros(size(g_w, 2), 1);
  for i = 1:model.q
    g_modelParams = g_modelParams + g_w(:, :, i)'*gX(:, i);
  end
  g = g_modelParams;
else
  % Do nothing
  g = gX';
end