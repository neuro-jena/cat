%-----------------------------------------------------------------------
% Job for longitudinal batch
% Christian Gaser
% $Id$
%-----------------------------------------------------------------------

global opts extopts output modulate dartel delete_temp ROImenu surfaces cat

if isempty(dartel),       dartel=0; end
if isempty(modulate),     modulate=1; end
if isempty(surfaces),     surfaces=cat_get_defaults('output.surfaces'); end
if isempty(delete_temp),  delete_temp=1; end


write_CSF = cat_get_defaults('output.CSF.mod') > 0;


if isfield(extopts,'admin') && isfield(extopts.admin,'lazy') && extopts.admin.lazy
  cat12('developer');
else
  cat12('expert');
end

% correct extopts fields for expert mode
tmp_fields = char('APP','LASstr','gcutstr','restypes');
segmentation = '';
for i=1:size(tmp_fields,1)
  segmentation = setfield(segmentation,deblank(tmp_fields(i,:)),extopts.(deblank(tmp_fields(i,:))));
  extopts = rmfield(extopts,deblank(tmp_fields(i,:)));
end
extopts.segmentation = segmentation;

warning('off','MATLAB:DELETE:FileNotFound');

% display start
if 0 %~isempty(extopts)  
  % The idea of simply repeat the input is not optimal.  
  % You have to use the DEP output otherwise it will result in more problems. 
  mbi = 1;
  matlabbatch{mbi}.cfg_basicio.run_ops.call_matlab.inputs{1}.images = '<UNDEFINED>';
  matlabbatch{mbi}.cfg_basicio.run_ops.call_matlab.outputs          = {}; % @(x) cat_io_depin2depout;
  matlabbatch{mbi}.cfg_basicio.run_ops.call_matlab.fun              = @(x)cat_io_cprintf('blue',sprintf([...
    '================================================================================================================================================\n' ...
    'Start CAT12 longitudinal processing of \n  %s\b\b\b\n' ...
    '================================================================================================================================================\n'],...
    sprintf('%s',char( cellfun(@(s) ([s(1:end-2) '\n  '])',x,'UniformOutput',0) )) ));
else
  mbi = 0;
end


% longitudinal rigid registration with final masking
%-----------------------------------------------------------------------
mbi = mbi + 1; mb_rigid = mbi; 
matlabbatch{mbi}.spm.tools.cat.tools.series.bparam          = 1e6;
matlabbatch{mbi}.spm.tools.cat.tools.series.use_brainmask   = 1;
matlabbatch{mbi}.spm.tools.cat.tools.series.reduce          = 1;
matlabbatch{mbi}.spm.tools.cat.tools.series.data            = '<UNDEFINED>';



% cat12 segmentation of average image 
%-----------------------------------------------------------------------
mbi = mbi + 1; mb_catavg = mbi;
matlabbatch{mbi}.spm.tools.cat.estwrite.data(1)             = cfg_dep('Longitudinal Registration: Midpoint Average', substruct('.','val', '{}',{mb_rigid}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','avg', '()',{':'}));
matlabbatch{mbi}.spm.tools.cat.estwrite.nproc               = 0;
if exist('opts','var') && ~isempty(opts)
	matlabbatch{mbi}.spm.tools.cat.estwrite.opts              = opts;
end
if exist('extopts','var') && ~isempty(extopts)
	matlabbatch{mbi}.spm.tools.cat.estwrite.extopts           = extopts;
end
if exist('output','var') && ~isempty(output)
	matlabbatch{mbi}.spm.tools.cat.estwrite.output            = output;
end
% surface estimation
matlabbatch{mbi}.spm.tools.cat.estwrite.output.surface      = surfaces;
if exist('ROImenu','var') && ~isempty(ROImenu)
  matlabbatch{mbi}.spm.tools.cat.estwrite.output.ROImenu    = ROImenu;
end
matlabbatch{mbi}.spm.tools.cat.estwrite.output.GM.native    = 0;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.GM.dartel    = 2;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.GM.mod       = 0;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.WM.native    = 0;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.WM.dartel    = 2;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.WM.mod       = 0;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.CSF.dartel   = 2;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.TPMC.dartel  = 2;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.bias.warped  = 0;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.warps        = [0 0];

% cat12 segmentation of realigned images 
%-----------------------------------------------------------------------
mbi = mbi + 1; mb_cat = mbi;
% use average image as prior for affine transformation and surface extraction
matlabbatch{mbi}.spm.tools.cat.estwrite.data(1)             = cfg_dep('Longitudinal Rigid Registration: Realigned images', substruct('.','val', '{}',{mb_rigid}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','rimg', '()',{':'}));
matlabbatch{mbi}.spm.tools.cat.estwrite.nproc               = 0;
if exist('opts','var') && ~isempty(opts)
	matlabbatch{mbi}.spm.tools.cat.estwrite.opts              = opts;
end
if exist('extopts','var') && ~isempty(extopts)
	matlabbatch{mbi}.spm.tools.cat.estwrite.extopts           = extopts;
end
if exist('output','var') && ~isempty(output)
	matlabbatch{mbi}.spm.tools.cat.estwrite.output            = output;
end
% surface estimation
matlabbatch{mbi}.spm.tools.cat.estwrite.output.surface      = surfaces;
if exist('ROImenu','var') && ~isempty(ROImenu)
  matlabbatch{mbi}.spm.tools.cat.estwrite.output.ROImenu    = ROImenu;
end
matlabbatch{mbi}.spm.tools.cat.estwrite.output.GM.native    = 1;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.GM.dartel    = dartel;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.GM.mod       = 0;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.WM.native    = 1;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.WM.dartel    = dartel;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.WM.mod       = 0;
if write_CSF
  matlabbatch{mbi}.spm.tools.cat.estwrite.output.CSF.native = 1; % also write CSF?
end
matlabbatch{mbi}.spm.tools.cat.estwrite.output.bias.warped  = 0;
matlabbatch{mbi}.spm.tools.cat.estwrite.output.warps        = [1 0];
matlabbatch{mbi}.spm.tools.cat.estwrite.useprior(1)         = cfg_dep('Longitudinal Registration: Midpoint Average', substruct('.','val', '{}',{mb_rigid}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','avg', '()',{':'}));

% averaging deformations
%-----------------------------------------------------------------------
mbi = mbi + 1; mb_avgdef = mbi;
matlabbatch{mbi}.spm.tools.cat.tools.avg_img.data(1)  = cfg_dep('CAT12: Segmentation (current release): Deformation Field', substruct('.','val', '{}',{mb_cat}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','fordef', '()',{':'}));
matlabbatch{mbi}.spm.tools.cat.tools.avg_img.output   = '';
matlabbatch{mbi}.spm.tools.cat.tools.avg_img.outdir   = {''};



% applying deformations to native segmentations
%-----------------------------------------------------------------------
mbi = mbi + 1; 
matlabbatch{mbi}.spm.tools.cat.tools.defs.field1(1)   = cfg_dep('Image Average: Average Image: ', substruct('.','val', '{}',{mb_avgdef}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{mbi}.spm.tools.cat.tools.defs.images(1)   = cfg_dep('CAT12: Segmentation (current release): p1 Image', substruct('.','val', '{}',{mb_cat}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','p', '()',{':'}));
matlabbatch{mbi}.spm.tools.cat.tools.defs.images(2)   = cfg_dep('CAT12: Segmentation (current release): p2 Image', substruct('.','val', '{}',{mb_cat}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','p', '()',{':'}));
if write_CSF
  matlabbatch{mbi}.spm.tools.cat.tools.defs.images(3) = cfg_dep('CAT12: Segmentation (current release): p3 Image', substruct('.','val', '{}',{mb_cat}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','p', '()',{':'}));
end
if modulate
  % modulation option for applying deformations
  matlabbatch{mbi}.spm.tools.cat.tools.defs.modulate  = modulate;
end
matlabbatch{mbi}.spm.tools.cat.tools.defs.interp      = 1;



% applying deformations to average T1 image
%-----------------------------------------------------------------------
mbi = mbi + 1; 
matlabbatch{mbi}.spm.tools.cat.tools.defs.field1(1)   = cfg_dep('Image Average: Average Image: ', substruct('.','val', '{}',{mb_avgdef}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{mbi}.spm.tools.cat.tools.defs.images(1)   = cfg_dep('Longitudinal Registration: Midpoint Average', substruct('.','val', '{}',{mb_rigid}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','avg', '()',{':'}));
matlabbatch{mbi}.spm.tools.cat.tools.defs.interp      = 1;
matlabbatch{mbi}.spm.tools.cat.tools.defs.modulate    = 0;



% delete temporary files
%-----------------------------------------------------------------------
if delete_temp
  mbi = mbi + 1; 
  matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(1)  = cfg_dep('CAT12: Segmentation (current release): p1 Image', substruct('.','val', '{}',{mb_cat}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','p', '()',{':'}));
  matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(2)  = cfg_dep('CAT12: Segmentation (current release): p2 Image', substruct('.','val', '{}',{mb_cat}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','p', '()',{':'}));
  matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(3)  = cfg_dep('CAT12: Segmentation (current release): Deformation Field', substruct('.','val', '{}',{mb_cat}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','fordef', '()',{':'}));
  matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(4)  = cfg_dep('CAT12: Segmentation (current release): rp1 affine Image', substruct('.','val', '{}',{mb_catavg}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','rpa', '()',{':'}));
  matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(5)  = cfg_dep('CAT12: Segmentation (current release): rp2 affine Image', substruct('.','val', '{}',{mb_catavg}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','rpa', '()',{':'}));
  matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(6)  = cfg_dep('CAT12: Segmentation (current release): rp3 affine Image', substruct('.','val', '{}',{mb_catavg}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','rpa', '()',{':'}));
  c = 7;
  if surfaces
    matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(c)  = cfg_dep('CAT12: Segmentation (current release): Left Central Surface', substruct('.','val', '{}',{mb_catavg}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','lhcentral', '()',{':'})); c = c+1;
    matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(c)  = cfg_dep('CAT12: Segmentation (current release): Left Thickness', substruct('.','val', '{}',{mb_catavg}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','lhthickness', '()',{':'})); c = c+1;
    matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(c)  = cfg_dep('CAT12: Segmentation (current release): Right Central Surface', substruct('.','val', '{}',{mb_catavg}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','rhcentral', '()',{':'})); c = c+1;
    matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(c)  = cfg_dep('CAT12: Segmentation (current release): Right Thickness', substruct('.','val', '{}',{mb_catavg}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','rhthickness', '()',{':'})); c = c+1;
  end
  if write_CSF
    matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.files(c) = cfg_dep('CAT12: Segmentation (current release): p3 Image', substruct('.','val', '{}',{mb_cat}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','p', '()',{':'}));
  end
  matlabbatch{mbi}.cfg_basicio.file_dir.file_ops.file_move.action.delete  = false;
end



% display finishing
%-----------------------------------------------------------------------
mbi = mbi + 1; 
matlabbatch{mbi}.cfg_basicio.run_ops.call_matlab.inputs{1}.images(1)  = cfg_dep('Longitudinal Rigid Registration: Realigned images', substruct('.','val', '{}',{mb_rigid}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','rimg', '()',{':'}));
matlabbatch{mbi}.cfg_basicio.run_ops.call_matlab.outputs              = {};
matlabbatch{mbi}.cfg_basicio.run_ops.call_matlab.fun                  = @(x)cat_io_cprintf('blue',sprintf([...
  '================================================================================================================================================\n' ...
  'Finished CAT12 longitudinal processing of \n  %s\b\b\b\n' ...
  '================================================================================================================================================\n'],...
  sprintf('%s',char( cellfun(@(s) ([s(1:end-2) '\n  '])',x,'UniformOutput',0) )) ));
