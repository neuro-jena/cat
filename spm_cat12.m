function spm_cat12(varargin)
% ______________________________________________________________________
% CAT12 Toolbox wrapper to call CAT functions.
% 
%   spm_cat12 
%     .. start with CAT default parameter file
%   spm_cat12('gui')
%     .. start with default file of another species (in development)
%   spm_cat12(species) 
%     .. start with default file of another species (in development)
%        species = ['oldwoldmonkey'|'newwoldmonkey'|'greaterape'|'lesserape']
%   spm_cat12('mypath/cat_defaults_mydefaults') 
%     .. start CAT with another default parameter file
% ______________________________________________________________________
% Christian Gaser
% $Id$

% ______________________________________________________________________
% Development:
%   spm_cat12('mypath/cat_defaults_mydefaults',1) 
%     .. restart SPM for GUI updates
% ______________________________________________________________________

rev = '$Rev$';

% start cat with different default file
catdir = fullfile(spm('dir'),'toolbox','cat12'); 
catdef = fullfile(catdir,'cat_defaults.m');
if nargin==0
  deffile = catdef; 
  restartspm = 0; 
else 
  deffile = varargin{1}; 
  restartspm = 1; 
end


% choose files
switch lower(deffile) 
  case {'select','choose'}
    deffile = spm_select(1,'batch','Select CAT default file!','',catdir);
  case 'gui'
    deffile = spm_input('Species class',1,'human|ape|monkey',...
      {'human','ape','monkey'},1);
    deffile = deffile{1}; 
    
    switch lower(deffile)
      %case 'human'
      %  deffile = spm_input('Species class','+1','adult|child|neonate|fetus|other',...
      %    {'human_adult','human_child','human_neonate','human_fetus','human_other'},1); 
      %  deffile = deffile{1};
      case 'ape'
        deffile = spm_input('Species class','+1','greater|lesser|other',...
          {'ape_greater','ape_lesser','other'},1);
        deffile = deffile{1};
      case 'monkey'
        deffile = spm_input('Species class','+1','old world|new world|other',...
          {'monkey_oldworld','monkey_newworld','other'},1);
        deffile = deffile{1};
    end
end

switch lower(deffile)
  case 'human'
    deffile = catdef; 
  case {'monkey_oldworld','oldwoldmonkey','cat_defaults_monkey_oldworld','cat_defaults_monkey_oldworld.m'}
    deffile = fullfile(catdir,'templates_animals','cat_defaults_monkey_oldworld.m');
  case {'monkey_newworld','newworldmonkey','cat_defaults_monkey_newworld','cat_defaults_monkey_newworld.m'}
    deffile = fullfile(catdir,'templates_animals','cat_defaults_monkey_newworld.m');
  case {'ape_greater','greaterape','cat_defaults_ape_greater','cat_defaults_ape_greater.m'}
    deffile = fullfile(catdir,'templates_animals','cat_defaults_ape_greater.m');
  case {'ape_lesser','lesserape','cat_defaults_ape_lesser','cat_defaults_ape_lesser.m'}
    deffile = fullfile(catdir,'templates_animals','cat_defaults_ape_lesser.m');
end

% lazy input - no extension 
[deffile_pp,deffile_ff,deffile_ee] = fileparts(deffile);
if isempty(deffile_ee)
  deffile_ee = '.m';
end
% lazy input - no directory
if isempty(deffile_pp) 
  if exist(fullfile(pwd,deffile_ff,deffile_ee),'file') 
    deffile_pp = pwd; 
  else
    deffile_pp = fullfile(spm('dir'),'toolbox','cat12'); 
  end
end
deffile = fullfile(deffile_pp,[deffile_ff,deffile_ee]); 

% check if file exist
if ~exist(deffile,'file')
  error('CAT:miss_cat_default_file','Can''t find CAT default file "%"','deffile'); 
end

% set other defaultfile
% The cat12 global variable is created and localy destroyed, because we 
% want to call the cat12 function. 
if 1 %nargin>0 %~strcmp(catdef,deffile) 
  oldwkd = cd; 
  cd(deffile_pp);
  try clearvars -global cat12; end
  clear cat12;
  eval(deffile_ff);
  eval('global cat12;'); 
  cd(oldwkd);
  
  % initialize SPM 
  eval('global defaults;'); 
  if isempty(defaults) || (nargin==2 && varargin{2}==1) || restartspm
    clear defaults; 
    spm_jobman('initcfg');
  end
  clear cat12;
end

SPMid = spm('FnBanner',mfilename,rev);
[Finter,Fgraph,CmdLine] = spm('FnUIsetup','CAT12');
url = fullfile(spm('Dir'),'toolbox','cat12','html','cat.html');
spm_help('!Disp',url,'',Fgraph,'Computational Anatomy Toolbox for SPM12');

% check whether CAT binaries will work
CATDir    = fullfile(spm('dir'),'toolbox','cat12','CAT');   
if ispc
  CATDir = [CATDir '.w32'];
elseif ismac
  CATDir = [CATDir '.maci64'];
elseif isunix
  CATDir = [CATDir '.glnx86'];
end  

[ST, RS] = system(fullfile(CATDir,'CAT_DumpCurv -h'));
% because status will not give 0 for help output we have to check whether we can find the
% keyword "Usage" in output
if isempty(findstr(RS,'Usage'));
  if ispc
    [ST, RS] = system('systeminfo.exe');
  else
    [ST, RS] = system('uname -a');
  end
  cat_io_cmd(sprintf('\nWARNING: Surface processing will not work because CAT-binaries are not compatible to your system:\n%s\n',RS),'warning');
  fprintf('\n\nFor future support of your system please send this message to christian.gaser@uni-jena.de\n\n');
end

if cat_get_defaults('extopts.gui')
  % command line output
  cat_io_cprintf([0.0 0.0 0.5],sprintf([ ...
    '\n' ...
    '   _______  ___  _______    \n' ...
    '  |  ____/ / _ \\\\ \\\\_   _/   \n' ...
    '  | |___  / /_\\\\ \\\\  | |     Computational Anatomy Toolbox\n' ...
    '  |____/ /_/   \\\\_\\\\ |_|     CAT12 - http://dbm.neuro.uni-jena.de\n\n']));
  cat_io_cprintf([0.0 0.0 0.5],sprintf([ ...
    ' CAT default file:\n' ...
    '\t%s\n\n'],deffile)); 

  % call GUI
  cat12('fig'); 
else
  fig = spm_figure('GetWin','Interactive');
  h0  = uimenu(fig,...
	'Label',	'CAT12',...
	'Separator',	'on',...
	'Tag',		'CAT',...
	'HandleVisibility','on');
  h1  = uimenu(h0,...
	'Label',	'Segment data',...
	'Separator',	'off',...
	'Tag',		'Estimate and write',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.estwrite'');',...
	'HandleVisibility','on');
  if 0
  h2  = uimenu(h0,...
	'Label',	'Write already estimated segmentations',...
	'Separator',	'off',...
	'Tag',		'Write segmentations',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.write'');',...
	'HandleVisibility','on');
  end
  if 0 % not yet ready
  h3  = uimenu(h0,...
	'Label',	'Process longitudinal data',...
	'Separator',	'off',...
	'Tag',		'Process longitudinal data',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.long'');',...
	'HandleVisibility','off');
  end
  h4  = uimenu(h0,...
	'Label',	'Check data quality',...
	'Separator',	'off',...
	'Tag',		'Check data quality',...
	'HandleVisibility','on');
  h41  = uimenu(h4,...
	'Label',	'Display one slice for all images',...
	'Separator',	'off',...
	'Tag',		'Display one slice for all images',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.showslice'');',...
	'HandleVisibility','on');
  h42  = uimenu(h4,...
	'Label',	'Check sample homogeneity using sample correlation',...
	'Separator',	'off',...
	'Tag',		'Check sample homogeneity using sample correlation',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.check_cov'');',...
	'HandleVisibility','on');
  if 0 % not yet ready
  h43  = uimenu(h4,...
	'Label',	'Check sample image quality',...
	'Separator',	'off',...
	'Tag',		'Check sample image quality',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.check_qa'');',...
	'HandleVisibility','on');
  end
  h5  = uimenu(h0,...
	'Label',	'Data presentation',...
	'Separator',	'off',...
	'Tag',		'Data presentation',...
	'HandleVisibility','on');
  h51  = uimenu(h5,...
	'Label',	'Calculate raw volumes for GM/WM/CSF',...
	'Separator',	'off',...
	'Tag',		'CAT Calculate raw volumes for GM/WM/CSF',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.calcvol'');',...
	'HandleVisibility','on');
  h52  = uimenu(h5,...
	'Label',	'Threshold and transform spmT-maps',...
	'Separator',	'off',...
	'Tag',		'Threshold and transform spmT-maps',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.T2x'');',...
	'HandleVisibility','on');
  h53  = uimenu(h5,...
	'Label',	'Threshold and transform spmF-maps',...
	'Separator',	'off',...
	'Tag',		'Threshold and transform spmF-maps',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.F2x'');',...
	'HandleVisibility','on');
  h54  = uimenu(h5,...
	'Label',	'Slice overlay',...
	'Separator',	'off',...
	'Tag',		'Slice overlay',...
	'CallBack','cat_vol_slice_overlay;',...
	'HandleVisibility','on');
  h6  = uimenu(h0,...
	'Label',	'Extended tools',...
	'Separator',	'off',...
	'Tag',		'Extended tools',...
	'HandleVisibility','on');
  h61  = uimenu(h6,...
	'Label',	'Spatial adaptive non local means denoising filter',...
	'Separator',	'off',...
	'Tag',		'SANLM filter',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.sanlm'');',...
	'HandleVisibility','on');
  h62  = uimenu(h6,...
	'Label',	'Intra-subject bias correction',...
	'Separator',	'off',...
	'Tag',		'Intra-subject bias correction',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.bias'');',...
	'HandleVisibility','on');
  if 0
  h63  = uimenu(h6,...
	'Label',	'Apply deformations (Many images)',...
	'Separator',	'off',...
	'Tag',		'Apply deformations (Many images)',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.defs'');',...
	'HandleVisibility','on');
  h64  = uimenu(h6,...
	'Label',	'Apply deformations (Many subjects)',...
	'Separator',	'off',...
	'Tag',		'Apply deformations (Many subjects)',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.defs2'');',...
	'HandleVisibility','on');
  end
  h65  = uimenu(h6,...
	'Label',	'Estimate origin using center-of-mass',...
	'Separator',	'off',...
	'Tag',		'Estimate origin using center-of-mass',...
	'CallBack','cat_vol_set_com;',...
	'HandleVisibility','on');
  h7  = uimenu(h0,...
	'Label',	'Surface tools',...
	'Separator',	'off',...
	'Tag',		'Surface tools',...
	'HandleVisibility','on');
  h71  = uimenu(h7,...
	'Label',	'Display surface',...
	'Separator',	'off',...
	'Tag',		'Display surface',...
	'CallBack', 'P=spm_select([1 24],''gifti'',''Select surface''); for i=1:size(P,1), h = spm_mesh_render(deblank(P(i,:))); set(h.figure,''MenuBar'',''none'',''Toolbar'',''none'',''Name'',spm_file(P(i,:),''short40''),''NumberTitle'',''off''); spm_mesh_render(''ColourMap'',h.axis,jet); spm_mesh_render(''ColourBar'',h.axis,''on'');end',...
	'HandleVisibility','on');
  h72  = uimenu(h7,...
	'Label',	'Extract surface parameters',...
	'Separator',	'off',...
	'Tag',		'Extract surface parameters',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.surfextract'');',...
	'HandleVisibility','on');
  h73  = uimenu(h7,...
	'Label',	'Resample and smooth surface parameters',...
	'Separator',	'off',...
	'Tag',		'Resample surface parameters to template space and smooth it',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.surfresamp'');',...
	'HandleVisibility','on');
  h74  = uimenu(h7,...
	'Label',	'Resample and smooth existing freesurfer thickness data',...
	'Separator',	'off',...
	'Tag',		'Resample existing freesurfer thickness data to template space and smooth it',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.surfresamp_fs'');',...
	'HandleVisibility','on');
  h75  = uimenu(h7,...
	'Label',	'Factorial design specification',...
	'Separator',	'off',...
	'Tag',		'Factorial design specification',...
	'CallBack','spm_jobman(''interactive'','''',''spm.stats.factorial_design'');',...
	'HandleVisibility','on');
  h76  = uimenu(h7,...
	'Label',	'Estimate design',...
	'Separator',	'off',...
	'Tag',		'Estimate design',...
	'CallBack','cat_stat_spm;',...
	'HandleVisibility','on');
  h77  = uimenu(h7,...
	'Label',	'Check sample homogeneity using sample correlation for surfaces',...
	'Separator',	'off',...
	'Tag',		'Check sample homogeneity using sample correlation for surfaces',...
	'CallBack','spm_jobman(''interactive'','''',''spm.tools.cat.tools.check_mesh_cov'');',...
	'HandleVisibility','on');
  h8  = uimenu(h0,...
	'Label',	'Print CAT debug information',...
	'Separator',	'on',...
	'Tag',		'Print debug information about versions and last error',...
	'CallBack','cat_debug;',...
	'HandleVisibility','on');
  h9  = uimenu(h0,...
	'Label',	'CAT Tools website',...
	'Separator',	'off',...
	'Tag',		'Launch CAT Tools site',...
	'CallBack',['set(gcbf,''Pointer'',''Watch''),',...
			'web(''http://dbm.neuro.uni-jena.de/cat12'',''-browser'');',...
			'set(gcbf,''Pointer'',''Arrow'')'],...
	'HandleVisibility','on');
  h10  = uimenu(h0,...
	'Label',	'Check for updates',...
	'Separator',	'off',...
	'Tag',		'Check for updates',...
	'CallBack','spm(''alert'',evalc(''cat_update(1)''),''CAT Update'');',...
	'HandleVisibility','on');
  h11  = uimenu(h0,...
	'Label',	'CAT12 Manual (PDF)',...
	'Separator',	'off',...
	'Tag',		'Open CAT12 Manual',...
	'CallBack','try,open(fullfile(spm(''dir''),''toolbox'',''cat12'',''CAT12-Manual.pdf''));end',...
	'HandleVisibility','on');

end
  
