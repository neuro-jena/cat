function spm_vbm8
% VBM5 Toolbox wrapper to call vbm functions
%_______________________________________________________________________
% Christian Gaser
% $Id$

rev = '$Rev$';

SPMid = spm('FnBanner',mfilename,rev);
[Finter,Fgraph,CmdLine] = spm('FnUIsetup','VBM8');
spm_help('!ContextHelp',mfilename);
spm_help('!Disp','vbm8.man','',Fgraph,'Voxel-based morphometry toolbox for SPM8');

fig = spm_figure('GetWin','Interactive');
h0  = uimenu(fig,...
	'Label',	'VBM8',...
	'Separator',	'on',...
	'Tag',		'VBM',...
	'HandleVisibility','on');
h1  = uimenu(h0,...
	'Label',	'Estimate and write',...
	'Separator',	'off',...
	'Tag',		'Estimate and write',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.estwrite'');',...
	'HandleVisibility','on');
h2  = uimenu(h0,...
	'Label',	'Write already estimated segmentations',...
	'Separator',	'off',...
	'Tag',		'Write segmentations',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.write'');',...
	'HandleVisibility','on');
h3  = uimenu(h0,...
	'Label',	'Process longitudinal data',...
	'Separator',	'off',...
	'Tag',		'Process longitudinal data',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.long'');',...
	'HandleVisibility','on');
h4  = uimenu(h0,...
	'Label',	'Check data quality',...
	'Separator',	'off',...
	'Tag',		'Check data quality',...
	'HandleVisibility','on');
h41  = uimenu(h4,...
	'Label',	'Display one slice for all images',...
	'Separator',	'off',...
	'Tag',		'Display one slice for all images',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.showslice'');',...
	'HandleVisibility','on');
h42  = uimenu(h4,...
	'Label',	'Check sample homogeneity using covariance',...
	'Separator',	'off',...
	'Tag',		'Check sample homogeneity using covariance',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.check_cov'');',...
	'HandleVisibility','on');
h5  = uimenu(h0,...
	'Label',	'Data presentation',...
	'Separator',	'off',...
	'Tag',		'Data presentation',...
	'HandleVisibility','on');
h51  = uimenu(h5,...
	'Label',	'Calculate raw volumes for GM/WM/CSF',...
	'Separator',	'off',...
	'Tag',		'VBM Calculate raw volumes for GM/WM/CSF',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.calcvol'');',...
	'HandleVisibility','on');
h52  = uimenu(h5,...
	'Label',	'Threshold and transform spmT-maps',...
	'Separator',	'off',...
	'Tag',		'Threshold and transform spmT-maps',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.T2x'');',...
	'HandleVisibility','on');
h53  = uimenu(h5,...
	'Label',	'Threshold and transform spmF-maps',...
	'Separator',	'off',...
	'Tag',		'Threshold and transform spmF-maps',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.F2x'');',...
	'HandleVisibility','on');
h54  = uimenu(h5,...
	'Label',	'Slice overlay',...
	'Separator',	'off',...
	'Tag',		'Slice overlay',...
	'CallBack','cg_slice_overlay;',...
	'HandleVisibility','on');
h6  = uimenu(h0,...
	'Label',	'Extended tools',...
	'Separator',	'off',...
	'Tag',		'Extended tools',...
	'HandleVisibility','on');
h54  = uimenu(h6,...
	'Label',	'Spatial adaptive non local means denoising filter',...
	'Separator',	'off',...
	'Tag',		'SANLM filter',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.sanlm'');',...
	'HandleVisibility','on');
h55  = uimenu(h6,...
	'Label',	'Intra-subject bias correction',...
	'Separator',	'off',...
	'Tag',		'Intra-subject bias correction',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.bias'');',...
	'HandleVisibility','on');
h56  = uimenu(h6,...
	'Label',	'Apply deformations',...
	'Separator',	'off',...
	'Tag',		'Apply deformations (Many images)',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.defs'');',...
	'HandleVisibility','on');
h57  = uimenu(h6,...
	'Label',	'Apply deformations',...
	'Separator',	'off',...
	'Tag',		'Apply deformations (Many subjects)',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.vbm8.tools.defs2'');',...
	'HandleVisibility','on');
h7  = uimenu(h0,...
	'Label',	'Print VBM debug information',...
	'Separator',	'on',...
	'Tag',		'Print debug information about versions and last error',...
	'CallBack','cg_vbm8_debug;',...
	'HandleVisibility','on');
h8  = uimenu(h0,...
	'Label',	'VBM Tools website',...
	'Separator',	'off',...
	'Tag',		'Launch VBM Tools site',...
	'CallBack',['set(gcbf,''Pointer'',''Watch''),',...
			'web(''http://dbm.neuro.uni-jena.de/vbm'',''-browser'');',...
			'set(gcbf,''Pointer'',''Arrow'')'],...
	'HandleVisibility','on');
h9  = uimenu(h0,...
	'Label',	'Check for updates',...
	'Separator',	'off',...
	'Tag',		'Check for updates',...
	'CallBack','cg_vbm8_update(1);',...
	'HandleVisibility','on');
h10  = uimenu(h0,...
	'Label',	'VBM8 Manual (PDF)',...
	'Separator',	'off',...
	'Tag',		'Open VBM8 Manual',...
	'CallBack','try,open(fullfile(spm(''dir''),''toolbox'',''vbm8'',''VBM8-Manual.pdf''));end',...
	'HandleVisibility','on');
