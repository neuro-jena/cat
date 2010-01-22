function cg_ornlm(vargin)
% 
% Optimized Blockwise Non Local Means Denoising Filter
%
%_______________________________________________________________________
% Christian Gaser
% $Id: cg_ornlm.m 224 2009-12-02 23:39:15Z gaser $

if nargin == 1
	P = [];
	for i=1:numel(vargin.data)
		P = strvcat(P,deblank(vargin.data{i}));
	end
else
  P = spm_select(Inf,'image','Select images to filter');
end

% get ORNLM weight
try
  ornlm_weight = spm_get_defaults('vbm8.extopts.ornlm');
catch
  ornlm_weight = spm_input('ORNLM weighting ?',1,'e',0.7);
end

V = spm_vol(P);
n = size(P,1);

spm_progress_bar('Init',n,'Filtering','Volumes Complete');
for i = 1:n
	[pth,nm,xt,vr] = fileparts(deblank(V(i).fname));
	in = spm_read_vols(V(i));
	h = rician_noise_estimation(in);

	if h>0
	  fprintf('Rician noise estimate for %s: %3.2f\n',nm,h);
  else
    h = gaussian_noise_estimation(in);
	  fprintf('Gaussian noise estimate for %s: %3.2f\n',nm,h);
  end

  % ORNLM weighting
  h = ornlm_weight*h;
  
  out = ornlmMex(in,3,1,h);
  V(i).fname = fullfile(pth,['ornlm_' nm xt vr]);
  V(i).descrip = sprintf('ORNLM filtered h=%3.2f',h);
  spm_write_vol(V(i), out);
	spm_progress_bar('Set',i);
end
spm_progress_bar('Clear');

return