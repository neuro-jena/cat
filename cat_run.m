function varargout = cat_run(job,arg)
% Segment a bunch of images
% FORMAT cat_run(job)
% job.channel(n).vols{m}
% job.channel(n).biasreg
% job.channel(n).biasfwhm
% job.channel(n).write
% job.tissue(k).tpm
% job.tissue(k).ngaus
% job.tissue(k).native
% job.tissue(k).warped
% job.cat.affreg
% job.cat.reg
% job.cat.samp
% job.cat.warps
% job.cat.darteltpm
% job.cat.print
%
% See the user interface for a description of the fields.
%_______________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% based on John Ashburners version of
% spm_preproc8_run.m 2281 2008-10-01 12:52:50Z john $
%
% Christian Gaser
% $Id$
%
%#ok<*AGROW>

%rev = '$Rev$';

  
% split job and data into separate processes to save computation time
if isfield(job,'nproc')
  if (job.nproc > 1) && (~isfield(job,'process_index'))

    fprintf('WARNING: Please note that no additional modules in the batch can be run except CAT12 Segmentation. The dependencies will be broken for any further modules if you split the job into separate processes.\n');

    % rescue original subjects
    job_data = job.data;
    n_subjects = numel(job.data);
    if job.nproc > n_subjects
      job.nproc = n_subjects;
    end
    job.process_index = cell(job.nproc,1);
  
    % initial splitting of data
    for i=1:job.nproc
      job.process_index{i} = (1:job.nproc:(n_subjects-job.nproc+1))+(i-1);
    end
  
    % check if all data are covered
    for i=1:rem(n_subjects,job.nproc)
      job.process_index{i} = [job.process_index{i} n_subjects-i+1];
    end
  
    tmp_array = cell(job.nproc,1);
    for i=1:job.nproc
      fprintf('Running job %d:\n',i);
      disp(job_data(job.process_index{i}));
      job.data = job_data(job.process_index{i});
    
      % temporary name for saving job information
      tmp_name = [tempname '.mat'];
      tmp_array{i} = tmp_name;
      save(tmp_name,'job');
    
      % matlab command          
      matlab_cmd = sprintf('"addpath %s %s %s %s;load %s; cat_run(job)"',spm('dir'),fullfile(spm('dir'),'toolbox','cat12'),...
          fullfile(spm('dir'),'toolbox','OldNorm'),fullfile(spm('dir'),'toolbox','DARTEL'), tmp_name);
    
      % log-file for output
      log_name = ['log' sprintf('%02d',i) '_' datestr(now,1) '_' strrep(datestr(now,15),':','_') '.txt'];
    
      % prepare system specific path for matlab
      export_cmd = ['set PATH=' fullfile(matlabroot,'bin')];
    
      % call matlab with command in the background
      if ispc
        system_cmd = [export_cmd ' & start matlab.bat -nodesktop -nosplash -r ' matlab_cmd ' -logfile ' log_name];
      else
        system_cmd = [export_cmd ';matlab -nodisplay -nosplash -r ' matlab_cmd ' -logfile ' log_name ' 2>&1 & '];
      end

      fprintf('Check %s  for logging information.\n',log_name);
      fprintf('_______________________________________________________________\n');
      [status,result] = system(system_cmd);
      
      % call editor for non-windows systems
      if ~ispc, edit(log_name); end
    end
        
    varargout{1} = [];
    return
  end
end

% check whether estimation & write should be done
estwrite = isfield(job,'opts');

% set some dummy defaults if segmentations are not estimated
if ~estwrite
    job.opts = struct('biasreg',0.001,'biasfwhm',60,'affreg','mni',...
                      'reg',[0 0.001 0.5 0.025 0.1],'samp',3,'ngaus',[3 3 2 3 4 2]);
end

channel = struct('vols',{job.data});

cat12 = struct('species', cat_get_defaults('extopts.species'), ... job.extopts.species,...
             'cat12atlas',cat_get_defaults('extopts.cat12atlas'), ... 
             'darteltpm', job.extopts.darteltpm{1}, ...
             'brainmask', cat_get_defaults('extopts.brainmask'), ...
             'affreg',    job.opts.affreg,...
             'samp',      cat_get_defaults('opts.samp'),...
             'warps',     job.output.warps,...
             'sanlm',     cat_get_defaults('extopts.sanlm'),... % job.extopts.sanlm,...
             'print',     job.extopts.print,...
             'ngaus',     cat_get_defaults('opts.ngaus'),...
             'reg',       cat_get_defaults('opts.warpreg'),...
             'bb',        cat_get_defaults('extopts.bb'),...
             'vox',       cat_get_defaults('extopts.vox'));

if isfield(job.extopts,'restype')
  cat12.restype = char(fieldnames(job.extopts.restype));
  cat12.resval  = job.extopts.restype.(cat12.restype); 
else
  cat12.restype = cat_get_defaults('extopts.restype');
  cat12.resval  = cat_get_defaults('extopts.resval');
end
if isfield(job.extopts,'sanlm')
  cat12.sanlm = job.extopts.sanlm;
end
if ~isfield(job.extopts,'verb')
  job.extopts.verb =  cat_get_defaults('extopts.verb');
end
if ~isfield(job.extopts,'APP')
  job.extopts.APP =  cat_get_defaults('extopts.APP');
end
if ~isfield(job.output,'ROI')
  job.output.ROI =  cat_get_defaults('output.ROI');
end
           
% set cat12.bb and vb.vox by Dartel template properties
Vd       = spm_vol([cat12.darteltpm ',1']);
[bb,vox] = spm_get_bbox(Vd, 'old');  
if cat12.bb(1)>cat12.bb(2), bbt=cat12.bb(1); cat12.bb(1)=cat12.bb(2); cat12.bb(2)=bbt; clear bbt; end
if bb(1)>bb(2), bbt=bb(1); bb(1)=bb(2); bb(2)=bbt; clear bbt; end
cat12.bb  = [ max(bb(1,1:3) , bb(1,1:3) ./ ((isinf(bb(1,1:3)) | isnan(bb(1,1:3)))+eps))
            min(bb(2,1:3) , bb(2,1:3) ./ ((isinf(bb(2,1:3)) | isnan(bb(2,1:3)))+eps)) ];
          
if isinf(cat12.vox) || isnan(cat12.vox)
  cat12.vox = abs(vox);
end



% prepare tissue priors and number of gaussians for all 6 classes
if estwrite
    [pth,nam,ext] = spm_fileparts(job.opts.tpm{1});
    clsn = numel(spm_vol(fullfile(pth,[nam ext]))); 
    tissue = struct();
    for i=1:clsn;
        tissue(i).ngaus = cat12.ngaus(i);
        tissue(i).tpm = [fullfile(pth,[nam ext]) ',' num2str(i)];
    end
end

% write tissue class 1-3              
tissue(1).warped = [job.output.GM.warped  (job.output.GM.modulated==1)  (job.output.GM.modulated==2) ];
tissue(1).native = [job.output.GM.native  (job.output.GM.dartel==1)     (job.output.GM.dartel==2)    ];
tissue(2).warped = [job.output.WM.warped  (job.output.WM.modulated==1)  (job.output.WM.modulated==2) ];
tissue(2).native = [job.output.WM.native  (job.output.WM.dartel==1)     (job.output.WM.dartel==2)    ];
if isfield(job.output,'CSF')
  tissue(3).warped = [job.output.CSF.warped (job.output.CSF.modulated==1) (job.output.CSF.modulated==2)];
  tissue(3).native = [job.output.CSF.native (job.output.CSF.dartel==1)    (job.output.CSF.dartel==2)   ];
else
  tissue(3).warped = [cat_get_defaults('output.CSF.warped') (cat_get_defaults('output.CSF.mod')==1)    (cat_get_defaults('output.CSF.mod')==2)];
  tissue(3).native = [cat_get_defaults('output.CSF.native') (cat_get_defaults('output.CSF.dartel')==1) (cat_get_defaults('output.CSF.dartel')==2)];
end

% never write class 4-6
for i=4:6;
    tissue(i).warped = [0 0 0];
    tissue(i).native = [0 0 0];
end

job.bias     = [job.output.bias.native  job.output.bias.warped job.output.bias.dartel];
if isfield(job.output,'label')
  job.label    = [job.output.label.native job.output.label.warped (job.output.label.dartel==1) (job.output.label.dartel==2)];
else
  job.label    = [cat_get_defaults('output.label.native') cat_get_defaults('output.label.warped') (cat_get_defaults('output.label.dartel')==1) (cat_get_defaults('output.label.dartel')==2)];
end
job.jacobian = job.output.jacobian.warped;

job.biasreg  = cat_get_defaults('opts.biasreg');
job.biasfwhm = cat_get_defaults('opts.biasfwhm');
job.channel  = channel;
job.cat      = cat12;
job.warps    = job.output.warps;
job.tissue   = tissue;

if nargin == 1, arg = 'run'; end

switch lower(arg)
    case 'run'
       varargout{1} = run_job(job,estwrite);
    case 'check'
        varargout{1} = check_job(job);
    case 'vfiles'
        varargout{1} = vfiles_job(job);
    case 'vout'
        varargout{1} = vout_job(job);
    otherwise
        error('Unknown argument ("%s").', arg);
end

return
%_______________________________________________________________________

%_______________________________________________________________________
function vout = run_job(job,estwrite)
  vout   = vout_job(job);

  if ~isfield(job.cat,'fwhm'),    job.cat.fwhm    =  1; end

  % load tpm priors only for estimate and write
  if estwrite
      tpm = char(cat(1,job.tissue(:).tpm));
      tpm = spm_load_priors8(tpm);
  else
      tpm = '';
  end

  for subj=1:numel(job.channel(1).vols),
    % __________________________________________________________________
    % Separation for old and new try-catch blocks of matlab. The new
    % try-catch block has to be in a separate file to avoid an error.
    % Both functions finally call cat_run_job.
    % See also cat_run_newcatch and cat_run_newcatch.
    % __________________________________________________________________
    matlabversion = version; 
    points = strfind(matlabversion,'.');
    if str2double(matlabversion(1:points(1)-1))<=7 && ...
       str2double(matlabversion(points(1)+1:points(2)-1))<=5
      cat_run_oldcatch(job,estwrite,tpm,subj);
    else
      cat_run_newcatch(job,estwrite,tpm,subj);
    end
  end

  colormap(gray)

return
%_______________________________________________________________________


%_______________________________________________________________________
function msg = check_job(job)
msg = {};
if numel(job.channel) >1,
    k = numel(job.channel(1).vols);
    for i=2:numel(job.channel),
        if numel(job.channel(i).vols)~=k,
            msg = {['Incompatible number of images in channel ' num2str(i)]};
            break
        end
    end
elseif numel(job.channel)==0,
    msg = {'No data'};
end
return
%_______________________________________________________________________

%_______________________________________________________________________
function vout = vout_job(job)

n     = numel(job.channel(1).vols);
parts = cell(n,4);

biascorr  = {};
wbiascorr = {};
label  = {};
wlabel = {};
rlabel = {};
alabel = {};
%jacobian = {};

for j=1:n,
    [parts{j,:}] = spm_fileparts(job.channel(1).vols{j});
end

if job.bias(1),
    biascorr = cell(n,1);
    for j=1:n
        biascorr{j} = fullfile(parts{j,1},['m',parts{j,2},'.nii']);
    end
end

if job.bias(2),
    wbiascorr = cell(n,1);
    for j=1:n
        wbiascorr{j} = fullfile(parts{j,1},['wm',parts{j,2},'.nii']);
    end
end

if job.label(1),
    label = cell(n,1);
    for j=1:n
        label{j} = fullfile(parts{j,1},['p0',parts{j,2},'.nii']);
    end
end

if job.label(2),
    wlabel = cell(n,1);
    for j=1:n
        wlabel{j} = fullfile(parts{j,1},['wp0',parts{j,2},'.nii']);
    end
end

if job.label(3),
    rlabel = cell(n,1);
    for j=1:n
        rlabel{j} = fullfile(parts{j,1},['rp0',parts{j,2},'.nii']);
    end
end

if job.label(4),
    alabel = cell(n,1);
    for j=1:n
        alabel{j} = fullfile(parts{j,1},['rp0',parts{j,2},'_affine.nii']);
    end
end

param = cell(n,1);
for j=1:n
    param{j} = fullfile(parts{j,1},['cat12_',parts{j,2},'.mat']);
end

tiss = struct('c',{},'rc',{},'rca',{},'wc',{},'mwc',{},'m0wc',{});
for i=1:numel(job.tissue),
    if job.tissue(i).native(1),
        tiss(i).c = cell(n,1);
        for j=1:n
            tiss(i).c{j} = fullfile(parts{j,1},['p',num2str(i),parts{j,2},'.nii']);
        end
    end
    if job.tissue(i).native(2),
        tiss(i).rc = cell(n,1);
        for j=1:n
            tiss(i).rc{j} = fullfile(parts{j,1},['rp',num2str(i),parts{j,2},'.nii']);
        end
    end
    if job.tissue(i).native(3),
        tiss(i).rca = cell(n,1);
        for j=1:n
            tiss(i).rca{j} = fullfile(parts{j,1},['rp',num2str(i),parts{j,2},'_affine.nii']);
        end
    end
    if job.tissue(i).warped(1),
        tiss(i).wc = cell(n,1);
        for j=1:n
            tiss(i).wc{j} = fullfile(parts{j,1},['wp',num2str(i),parts{j,2},'.nii']);
        end
    end
    if job.tissue(i).warped(2),
        tiss(i).mwc = cell(n,1);
        for j=1:n
            tiss(i).mwc{j} = fullfile(parts{j,1},['mwp',num2str(i),parts{j,2},'.nii']);
        end
    end
    if job.tissue(i).warped(3),
        tiss(i).m0wc = cell(n,1);
        for j=1:n
            tiss(i).m0wc{j} = fullfile(parts{j,1},['m0wp',num2str(i),parts{j,2},'.nii']);
        end
    end
end

if job.cat.warps(1),
    fordef = cell(n,1);
    for j=1:n
        fordef{j} = fullfile(parts{j,1},['y_',parts{j,2},'.nii']);
    end
else
    fordef = {};
end

if job.cat.warps(2),
    invdef = cell(n,1);
    for j=1:n
        invdef{j} = fullfile(parts{j,1},['iy_',parts{j,2},'.nii']);
    end
else
    invdef = {};
end

if job.jacobian,
    jacobian = cell(n,1);
    for j=1:n
        jacobian{j} = '';
    end
else
    jacobian = {};
end

vout  = struct('tiss',tiss,'label',{label},'wlabel',{wlabel},'rlabel',{rlabel},'alabel',{alabel},...
               'biascorr',{biascorr},'wbiascorr',{wbiascorr},'param',{param},...
               'invdef',{invdef},'fordef',{fordef},'jacobian',{jacobian});
%_______________________________________________________________________

%_______________________________________________________________________
function vf = vfiles_job(job)
vout = vout_job(job);
vf   = vout.param;
if ~isempty(vout.invdef),     vf = [vf vout.invdef]; end
if ~isempty(vout.fordef),     vf = [vf, vout.fordef]; end
if ~isempty(vout.jacobian),   vf = [vf, vout.jacobian]; end

if ~isempty(vout.biascorr),   vf = [vf, vout.biascorr]; end
if ~isempty(vout.wbiascorr),  vf = [vf, vout.wbiascorr]; end
if ~isempty(vout.label),      vf = [vf, vout.label]; end
if ~isempty(vout.wlabel),     vf = [vf, vout.wlabel]; end
if ~isempty(vout.rlabel),     vf = [vf, vout.rlabel]; end
if ~isempty(vout.alabel),     vf = [vf, vout.alabel]; end

for i=1:numel(vout.tiss)
    if ~isempty(vout.tiss(i).c),   vf = [vf vout.tiss(i).c];   end 
    if ~isempty(vout.tiss(i).rc),  vf = [vf vout.tiss(i).rc];  end 
    if ~isempty(vout.tiss(i).rca), vf = [vf vout.tiss(i).rca]; end
    if ~isempty(vout.tiss(i).wc),  vf = [vf vout.tiss(i).wc];  end
    if ~isempty(vout.tiss(i).mwc), vf = [vf vout.tiss(i).mwc]; end
    if ~isempty(vout.tiss(i).m0wc),vf = [vf vout.tiss(i).m0wc];end
end
vf = reshape(vf,numel(vf),1);
%_______________________________________________________________________

%=======================================================================
