function vbm8 = tbx_cfg_vbm8
% Configuration file for Segment jobs
%_______________________________________________________________________
% Copyright (C) 2008 Wellcome Department of Imaging Neuroscience

% based on John Ashburners version of
% tbx_cfg_preproc8.m
%
% Christian Gaser
% $Id$

rev = '$Rev$';

cg_vbm8_defaults

addpath(fileparts(which(mfilename)));

%_______________________________________________________________________

data = cfg_files;
data.tag  = 'data';
data.name = 'Volumes';
data.help = {[...
'Select raw data (e.g. T1 images) for processing. ',...
'This assumes that there is one scan for each subject. ',...
'Note that multi-spectral (when there are two or more registered ',...
'images of different contrasts) processing is not yet implemented ',...
'for this method.']};
data.filter = 'image';
data.ufilter = '.*';
data.num     = [1 Inf];

%------------------------------------------------------------------------
% various options for estimating the segmentations
%------------------------------------------------------------------------

ngaus      = cfg_entry;
ngaus.tag  = 'ngaus';
ngaus.name = 'Gaussians per class';
ngaus.strtype = 'e';
ngaus.num = [1 6];
ngaus.def  = @(val)spm_get_defaults('vbm8.opts.ngaus', val{:});
ngaus.help = {[...
'The number of Gaussians used to represent the intensity distribution '...
'for each tissue class can be greater than one. '...
'In other words, a tissue probability map may be shared by several clusters. '...
'The assumption of a single Gaussian distribution for each class does not '...
'hold for a number of reasons. '...
'In particular, a voxel may not be purely of one tissue type, and instead '...
'contain signal from a number of different tissues (partial volume effects). '...
'Some partial volume voxels could fall at the interface between different '...
'classes, or they may fall in the middle of structures such as the thalamus, '...
'which may be considered as being either grey or white matter. '...
'Various other image segmentation approaches use additional clusters to '...
'model such partial volume effects. '...
'These generally assume that a pure tissue class has a Gaussian intensity '...
'distribution, whereas intensity distributions for partial volume voxels '...
'are broader, falling between the intensities of the pure classes. '...
'Unlike these partial volume segmentation approaches, the model adopted '...
'here simply assumes that the intensity distribution of each class may '...
'not be Gaussian, and assigns belonging probabilities according to these '...
'non-Gaussian distributions. '...
'Typical numbers of Gaussians could be two for grey matter, two for white '...
'matter, two for CSF, three for bone, four for other soft tissues and ',...
'two for air (background).'],[...
'Note that if any of the Num. Gaussians is set to non-parametric, '...
'then a non-parametric approach will be used to model the tissue intensities. '...
'This may work for some images (eg CT), but not others - '...
'and it has not been optimised for multi-channel data. Note that it is likely to ',...
'be especially problematic for images with poorly behaved intensity histograms ',...
'due to aliasing effects that arise from having discrete values on the images.']};

%------------------------------------------------------------------------

biasreg = cfg_menu;
biasreg.tag  = 'biasreg';
biasreg.name = 'Bias regularisation';
biasreg.def  = @(val)spm_get_defaults('vbm8.opts.biasreg', val{:});
biasreg.labels = {...
'no regularisation (0)','extremely light regularisation (0.00001)',...
'very light regularisation (0.0001)','light regularisation (0.001)',...
'medium regularisation (0.01)','heavy regularisation (0.1)',...
'very heavy regularisation (1)','extremely heavy regularisation (10)'};
biasreg.values = {0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10};
biasreg.help = {[...
'MR images are usually corrupted by a smooth, spatially varying artifact that modulates the intensity ',...
'of the image (bias). ',...
'These artifacts, although not usually a problem for visual inspection, can impede automated ',...
'processing of the images.'],...
'',...
[...
'An important issue relates to the distinction between intensity variations that arise because of ',...
'bias artifact due to the physics of MR scanning, and those that arise due to different tissue ',...
'properties.  The objective is to model the latter by different tissue classes, while modelling the ',...
'former with a bias field. ',...
'We know a priori that intensity variations due to MR physics tend to be spatially smooth, ',...
'whereas those due to different tissue types tend to contain more high frequency information. ',...
'A more accurate estimate of a bias field can be obtained by including prior knowledge about ',...
'the distribution of the fields likely to be encountered by the correction algorithm. ',...
'For example, if it is known that there is little or no intensity non-uniformity, then it would be wise ',...
'to penalise large values for the intensity non-uniformity parameters. ',...
'This regularisation can be placed within a Bayesian context, whereby the penalty incurred is the negative ',...
'logarithm of a prior probability for any particular pattern of non-uniformity.'],...
['Knowing what works best should be a matter '...
'of empirical exploration.  For example, if your data has very little '...
'intensity non-uniformity artifact, then the bias regularisation should '...
'be increased.  This effectively tells the algorithm that there is very little '...
'bias in your data, so it does not try to model it.']};

%------------------------------------------------------------------------

biasfwhm    = cfg_menu;
biasfwhm.tag = 'biasfwhm';
biasfwhm.name = 'Bias FWHM';
biasfwhm.labels = {...
'30mm cutoff','40mm cutoff','50mm cutoff','60mm cutoff','70mm cutoff',...
'80mm cutoff','90mm cutoff','100mm cutoff','110mm cutoff','120mm cutoff',...
'130mm cutoff','140mm cutoff','150mm cutoff','No correction'};
biasfwhm.values = {30,40,50,60,70,80,90,100,110,120,130,140,150,Inf};
biasfwhm.def  = @(val)spm_get_defaults('vbm8.opts.biasfwhm', val{:});
biasfwhm.help = {[...
'FWHM of Gaussian smoothness of bias. ',...
'If your intensity non-uniformity is very smooth, then choose a large ',...
'FWHM. This will prevent the algorithm from trying to model out intensity variation ',...
'due to different tissue types. The model for intensity non-uniformity is one ',...
'of i.i.d. Gaussian noise that has been smoothed by some amount, ',...
'before taking the exponential. ',...
'Note also that smoother bias fields need fewer parameters to describe them. ',...
'This means that the algorithm is faster for smoother intensity non-uniformities.']};

%------------------------------------------------------------------------

warpreg      = cfg_entry;
warpreg.def  = @(val)spm_get_defaults('vbm8.opts.warpreg', val{:});
warpreg.tag = 'warpreg';
warpreg.name = 'Warping Regularisation';
warpreg.strtype = 'e';
warpreg.num = [1 1];
warpreg.help = {[...
'The objective function for registering the tissue probability maps to the ',...
'image to process, involves minimising the sum of two terms. ',...
'One term gives a function of how probable the data is given the warping parameters. ',...
'The other is a function of how probable the parameters are, and provides a ',...
'penalty for unlikely deformations. ',...
'Smoother deformations are deemed to be more probable. ',...
'The amount of regularisation determines the tradeoff between the terms. ',...
'Pick a value around one.  However, if your normalised images appear ',...
'distorted, then it may be an idea to increase the amount of ',...
'regularisation (by an order of magnitude). ',...
'More regularisation gives smoother deformations, ',...
'where the smoothness measure is determined by the bending energy of the deformations. ']};
%------------------------------------------------------------------------

affreg = cfg_menu;
affreg.tag = 'affreg';
affreg.name = 'Affine Regularisation';
affreg.labels = {'No Affine Registration','ICBM space template - European brains',...
    'ICBM space template - East Asian brains', 'Average sized template','No regularisation'};
affreg.values = {'','mni','eastern','subj','none'};
affreg.def  = @(val)spm_get_defaults('vbm8.opts.affreg', val{:});
affreg.help = {[...
'The procedure is a local optimisation, so it needs reasonable initial '...
'starting estimates. Images should be placed in approximate alignment '...
'using the Display function of SPM before beginning. '...
'A Mutual Information affine registration with the tissue '...
'probability maps (D''Agostino et al, 2004) is used to achieve '...
'approximate alignment. '...
'Note that this step does not include any model for intensity non-uniformity. '...
'This means that if the procedure is to be initialised with the affine '...
'registration, then the data should not be too corrupted with this artifact.'...
'If there is a lot of intensity non-uniformity, then manually position your '...
'image in order to achieve closer starting estimates, and turn off the '...
'affine registration.'],...
'',...
[...
'Affine registration into a standard space can be made more robust by ',...
'regularisation (penalising excessive stretching or shrinking).  The ',...
'best solutions can be obtained by knowing the approximate amount of ',...
'stretching that is needed (e.g. ICBM templates are slightly bigger ',...
'than typical brains, so greater zooms are likely to be needed). ',...
'For example, if registering to an image in ICBM/MNI space, then choose this ',...
'option.  If registering to a template that is close in size, then ',...
'select the appropriate option for this.']};

%------------------------------------------------------------------------

affmethod = cfg_menu;
affmethod.tag = 'affmethod';
affmethod.name = 'Affine Registration Method';
affmethod.labels = {...
    'Seg Default (mutual information)',...
    'Least Squares with masked T1 template'};
affmethod.values = {0, 1};
affmethod.def  = @(val)spm_get_defaults('vbm8.opts.affmethod', val{:});
affmethod.help = {[...
'An initial affine registration is neccessary to register images ',...
'to MNI space. As default this registration is based on mutual information and ',...
'the tissue priors are used as template. However, sometimes this registration fails ',...
'and as alternative the standard affine registration based on least squares with the ',...
'T1 template and a template mask can be choosen. For T1 images this is the preferable method.']};

%------------------------------------------------------------------------

samp      = cfg_entry;
samp.tag = 'samp';
samp.name = 'Sampling distance';
samp.strtype = 'e';
samp.num = [1 1];
samp.val  = {3};
samp.help = {[...
'The approximate distance between sampled points when estimating the ',...
'model parameters. Smaller values use more of the data, but the procedure ',...
'is slower.']};

%------------------------------------------------------------------------

opts      = cfg_branch;
opts.tag = 'opts';
opts.name = 'Estimation options';
opts.val = {ngaus,biasreg,biasfwhm,affmethod,affreg,warpreg,samp};
opts.help = {[...
'Various options can be adjusted in order to improve the performance of the ',...
'algorithm with your data.  Knowing what works best should be a matter ',...
'of empirical exploration. For example, if your data has very little ',...
'intensity nonuniformity artifact, then the bias regularisation should ',...
'be increased. This effectively tells the algorithm that there is very little ',...
'bias in your data, so it does not try to model it.']};

%_______________________________________________________________________
% options for output
%-----------------------------------------------------------------------

bb      = cfg_entry;
bb.tag = 'bb';
bb.name = 'Bounding box';
bb.strtype = 'e';
bb.num = [2 3];
bb.def  = @(val)spm_get_defaults('vbm8.extopts.bb', val{:});
bb.help = {[...
'The bounding box (in mm) of any spatially normalised volumes to be written ',...
'(relative to the anterior commissure). '...
'Non-finite values will be replaced by the bounding box of the tissue '...
'probability maps used in the segmentation.']};

%------------------------------------------------------------------------

vox      = cfg_entry;
vox.tag = 'vox';
vox.name = 'Voxel size';
vox.strtype = 'e';
vox.num = [1 1];
vox.def  = @(val)spm_get_defaults('vbm8.extopts.vox', val{:});
vox.help = {...
['The (isotropic) voxel sizes of any spatially normalised written images. '...
 'A non-finite value will be replaced by the average voxel size of '...
 'the tissue probability maps used by the segmentation.']};

%------------------------------------------------------------------------

cleanup = cfg_menu;
cleanup.tag  = 'cleanup';
cleanup.name = 'Clean up any partitions';
cleanup.help = {[...
'This uses a crude routine for extracting the brain from segmented',...
'images. It begins by taking the white matter, and eroding it a',...
'couple of times to get rid of any odd voxels. The algorithm',...
'continues on to do conditional dilations for several iterations,',...
'where the condition is based upon gray or white matter being present.',...
'This identified region is then used to clean up the grey and white',...
'matter partitions, and has a slight influences on the CSF partition.'],'',[...
'If you find pieces of brain being chopped out in your data, then you ',...
'may wish to disable or tone down the cleanup procedure.']};
cleanup.labels = {'Dont do cleanup','Light Clean','Thorough Clean'};
cleanup.values = {0 1 2};
cleanup.def  = @(val)spm_get_defaults('vbm8.extopts.cleanup', val{:});

%------------------------------------------------------------------------

brainmask = cfg_files;
brainmask.tag = 'brainmask';
brainmask.name = 'Brainmask for skull stripping';
brainmask.dir = fileparts(which(mfilename));
brainmask.filter = 'image';
brainmask.ufilter = '.*';
brainmask.num = [1 1];
brainmask.def  = @(val)spm_get_defaults('vbm8.extopts.brainmask', val{:});
brainmask.help = {[...
'The segmentation should be restricted to intracranial parts of the brain and ',...
'therefore the skull and scalp have to be removed from the images. In SPM this is ',...
'provided with a Bayesian approach of using tissue priors with up to 6 classes to ',...
'remove background, scalp, and skull. For the current approach we do not rely on ',...
'tissue priors and images have to be skull-stripped before the segmentation. This is ',...
'achieved by using a pedefined brainmask, which is based on the warped average ',...
'of 40 manually stripped brains from the LPBA40 data set from the Laboratory of Neuroimaging at UCLA ',...
'(http://www.loni.ucla.edu/Atlases/Atlas_Detail.jsp?atlas_id=12).']};

%------------------------------------------------------------------------

brainmask_th      = cfg_entry;
brainmask_th.tag = 'brainmask_th';
brainmask_th.name = 'Threshold for brainmask';
brainmask_th.strtype = 'e';
brainmask_th.num = [1 1];
brainmask_th.def  = @(val)spm_get_defaults('vbm8.extopts.brainmask_th', val{:});
brainmask_th.help = {...
['The default threshold is 0.5, which means that all areas, where the probability '...
 'of the brainmask is > 50% are used as mask. This default threshold works very well '...
 'for most brains. In case that parts of GM are also removed you could use a lower '...
 'threshold (e.g. 0.25..0.5).']};

%------------------------------------------------------------------------

print    = cfg_menu;
print.tag = 'print';
print.name = 'Display and print results';
print.labels = {'yes','no'};
print.values = {0 1};
print.def  = @(val)spm_get_defaults('vbm8.extopts.print', val{:});
print.help = {[...
'The normalized T1 image and the normalized segmentations can be displayed and printed to a ',...
'ps-file. This is often helpful to check whether registration and segmentation were successful.']};

%------------------------------------------------------------------------

extopts      = cfg_branch;
extopts.tag = 'extopts';
extopts.name = 'Extended options';
extopts.val = {bb,brainmask,brainmask_th};
extopts.help = {'Extended options'};

%------------------------------------------------------------------------
% options for data
%------------------------------------------------------------------------

native    = cfg_menu;
native.tag = 'native';
native.name = 'Native space';
native.labels = {'none','yes'};
native.values = {0 1};
native.help = {'Write image in native space.'};

warped    = cfg_menu;
warped.tag = 'warped';
warped.name = 'Normalized';
warped.labels = {'none','yes'};
warped.values = {0 1};
warped.help = {'Write image in normalized space.'};

native.def  = @(val)spm_get_defaults('vbm8.output.bias.native', val{:});
warped.def  = @(val)spm_get_defaults('vbm8.output.bias.warped', val{:});
bias      = cfg_branch;
bias.tag = 'bias';
bias.name = 'Bias Corrected';
bias.val = {native, warped};
bias.help = {[...
'This is the option to save a bias corrected version of your image. ',...
'MR images are usually corrupted by a smooth, spatially varying artifact that modulates the intensity ',...
'of the image (bias). ',...
'These artifacts, although not usually a problem for visual inspection, can impede automated ',...
'processing of the images. The bias corrected version should have more uniform intensities within ',...
'the different types of tissues and can be saved in native space and/or normalised.']};

%------------------------------------------------------------------------

native.def  = @(val)spm_get_defaults('vbm8.output.label.native', val{:});
warped.def  = @(val)spm_get_defaults('vbm8.output.label.warped', val{:});
label      = cfg_branch;
label.tag = 'label';
label.name = 'PVE label image';
label.val = {native, warped};
label.help = {[...
'This is the option to save a labeled version of your segmentations. ',...
'Labels are saved as PVE values.']};

%------------------------------------------------------------------------

modulated    = cfg_menu;
modulated.tag = 'modulated';
modulated.name = 'Modulated normalized';
modulated.labels = {'none','affine + non-linear (SPM8 default)','non-linear only'};
modulated.values = {0 1 2};
modulated.help = {[...
'Modulation is to compensate for the effect of spatial normalisation. Spatial normalisation ',...
'causes volume changes due to affine transformation (global scaling) and non-linear warping (local volume change). ',...
'The SPM default is to adjust spatially normalised grey matter (or other tissue class) by using both terms and the ',...
'resulting modulated images are preserved for the total amount of grey matter. Thus, modulated images reflect the grey matter ',...
'volumes before spatial normalisation. However, the user is often interested in removing the confound of different brain sizes ',...
'and there are many ways to apply this correction. We can use the total amount of GM, GM+WM, GM+WM+CSF, or manual estimated ',...
'total intracranial volume (TIV). Theses parameters can be modeled as nuisance parameters (additive effects) in an AnCova model ',...
'or used to globally scale the data (multiplicative effects): '],...
'',...
'% Correction   Interpretation',...
'% ----------   --------------',...
'% nothing      absolute volume',...
'% globals 	     relative volume after correcting for total GM or TIV (multiplicative effects)',...
'% AnCova 	      relative volume that can not be explained by total GM or TIV (additive effects)',...
'',...
[...
'I suggest another option to remove the confounding effects of different brain sizes. Modulated images can be optionally saved ',...
'by correcting for non-linear warping only. Volume changes due to affine normalisation will be not considered and this equals ',...
'the use of default modulation and globally scaling data according to the inverse scaling factor due to affine normalisation. I recommend ',...
'this option if your hypothesis is about effects of relative volumes which are corrected for different brain sizes. This is a widely ',...
'used hypothesis and should fit to most data. The idea behind this option is that scaling of affine normalisation is indeed a ',...
'multiplicative (gain) effect and we rather apply this correction to our data and not to our statistical model. ',...
'These modulated images are indicated by "m0" instead of "m". ']};

dartel    = cfg_menu;
dartel.tag = 'dartel';
dartel.name = 'DARTEL export';
dartel.labels = {'none','rigid (SPM8 default)','affine'};
dartel.values = {0 1 2};
dartel.help = {['This option is to export data into a form that can be used with DARTEL.',...
'The SPM8 default is to only apply rigid body transformation. An additional option is to ',...
'apply affine transformation.']};

native.def    = @(val)spm_get_defaults('vbm8.output.grey.native', val{:});
warped.def    = @(val)spm_get_defaults('vbm8.output.grey.warped', val{:});
modulated.def = @(val)spm_get_defaults('vbm8.output.grey.mod', val{:});
dartel.def    = @(val)spm_get_defaults('vbm8.output.grey.dartel', val{:});
grey      = cfg_branch;
grey.tag = 'GM';
grey.name = 'Grey matter';
grey.val = {native, warped, modulated, dartel};
grey.help     = {'Options to produce grey matter images: p1*.img, wp1*.img and mwp1*.img.'};

native.def    = @(val)spm_get_defaults('vbm8.output.white.native', val{:});
warped.def    = @(val)spm_get_defaults('vbm8.output.white.warped', val{:});
modulated.def = @(val)spm_get_defaults('vbm8.output.white.mod', val{:});
dartel.def    = @(val)spm_get_defaults('vbm8.output.white.dartel', val{:});
white      = cfg_branch;
white.tag = 'WM';
white.name = 'White matter';
white.val = {native, warped, modulated, dartel};
white.help    = {'Options to produce white matter images: p2*.img, wp2*.img and mwp2*.img.'};

native.def    = @(val)spm_get_defaults('vbm8.output.csf.native', val{:});
warped.def    = @(val)spm_get_defaults('vbm8.output.csf.warped', val{:});
modulated.def = @(val)spm_get_defaults('vbm8.output.csf.mod', val{:});
dartel.def    = @(val)spm_get_defaults('vbm8.output.csf.dartel', val{:});
csf      = cfg_branch;
csf.tag = 'CSF';
csf.name = 'Cerebro-Spinal Fluid (CSF)';
csf.val = {native, warped, modulated, dartel};
csf.help      = {'Options to produce CSF images: p3*.img, wp3*.img and mwp3*.img.'};

%------------------------------------------------------------------------

warps = cfg_menu;
warps.tag = 'warps';
warps.name = 'Deformation Fields';
warps.labels = {...
    'None',...
    'Inverse',...
    'Forward',...
    'Inverse + Forward'};
warps.values = {[0 0],[1 0],[0 1],[1 1]};
warps.def  = @(val)spm_get_defaults('vbm8.output.warps', val{:});
warps.help = {'Deformation fields can be written.'};

%------------------------------------------------------------------------

output      = cfg_branch;
output.tag = 'output';
output.name = 'Writing options';
output.val = {grey, white, csf, bias, label, warps};
output.help = {[...
'This routine produces spatial normalisation parameters (*_seg8_sn.mat files) by default. '],...
'',...
[...
'In addition, it also produces files that can be used for doing inverse normalisation. ',...
'If you have an image of regions defined in the standard space, then the inverse deformations ',...
'can be used to warp these regions so that it approximately overlay your image. ',...
'To use this facility, the bounding-box and voxel sizes should be set to non-finite values ',...
'(e.g. [NaN NaN NaN] for the voxel sizes, and ones(2,3)*NaN for the bounding box. ',...
'This would be done by the spatial normalisation module, which allows you to select a ',...
'set of parameters that describe the nonlinear warps, and the images that they should be applied to.'],...
'',...
[...
'There are a number of options about what data you would like the routine to produce. ',...
'The routine can be used for producing images of tissue classes, as well as bias corrected images. ',...
'The native space option will produce a tissue class image (c*) that is in alignment with ',...
'the original/* (see Figure \ref{seg1})*/. You can also produce spatially normalised versions - both',...
'with (mwc*) and without (wc*) modulation/* (see Figure \ref{seg2})*/. In contrast to Johns version',...
'the voxel sizes of the spatially normalised versions can be choosen, an HMRF can be applied, and the',...
'use of priors with the Bayes rule can be omitted.',...
'These can be used for doing voxel-based morphometry with (both un-modulated and modulated). ',...
'All you need to do is smooth them and do the stats (which means no more questions on the mailing list ',...
'about how to do "optimized VBM").'],...
'',...
[...
'Modulation is to compensate for the effect of spatial normalisation. When warping a series ',...
'of images to match a template, it is inevitable that volumetric differences will be introduced ',...
'into the warped images. For example, if one subject''s temporal lobe has half the volume of that of ',...
'the template, then its volume will be doubled during spatial normalisation. This will also ',...
'result in a doubling of the voxels labeled grey matter. In order to remove this confound, the ',...
'spatially normalised grey matter (or other tissue class) is adjusted by multiplying by its relative ',...
'volume before and after warping. If warping results in a region doubling its volume, then the ',...
'correction will halve the intensity of the tissue label. This whole procedure has the effect of preserving ',...
'the total amount of grey matter signal in the normalised partitions.'],...
['/*',...
'\begin{figure} ',...
'\begin{center} ',...
'\includegraphics[width=150mm]{images/seg1} ',...
'\end{center} ',...
'\caption{Segmentation results. ',...
'These are the results that can be obtained in the original space of the image ',...
'(i.e. the results that are not spatially normalised). ',...
'Top left: original image (X.img). ',...
'Top right: bias corrected image (mX.img). ',...
'Middle and bottom rows: segmented grey matter (c1X.img), ',...
'white matter (c2X.img) and CSF (c3X.img). \label{seg1}} ',...
'\end{figure} */'],...
['/*',...
'\begin{figure} ',...
'\begin{center} ',...
'\includegraphics[width=150mm]{images/seg2} ',...
'\end{center} ',...
'\caption{Segmentation results. ',...
'These are the spatially normalised results that can be obtained ',...
'(note that CSF data is not shown). ',...
'Top row: The tissue probability maps used to guide the segmentation. ',...
'Middle row: Spatially normalised tissue maps of grey and white matter ',...
'(wc1X.img and wc2X.img). ',...
'Bottom row: Modulated spatially normalised tissue maps of grey and ',...
'white matter (mwc1X.img and mwc2X.img). \label{seg2}} ',...
'\end{figure} */'],...
[...
'A deformation field is a vector field, where three values are associated with ',...
'each location in the field. The field maps from co-ordinates in the ',...
'normalised image back to co-ordinates in the original image. The value of ',...
'the field at co-ordinate [x y z] in the normalised space will be the ',...
'co-ordinate [x'' y'' z''] in the original volume. ',...
'The gradient of the deformation field at a co-ordinate is its Jacobian ',...
'matrix, and it consists of a 3x3 matrix:'],...
'',...
'%   /                      \',...
'%   | dx''/dx  dx''/dy dx''/dz |',...
'%   |                       |',...
'%   | dy''/dx  dy''/dy dy''/dz |',...
'%   |                       |',...
'%   | dz''/dx  dz''/dy dz''/dz |',...
'%   \                      /',...
['/* \begin{eqnarray*}',...
'\begin{pmatrix}',...
'\frac{dx''}{dx} & \frac{dx''}{dy} & \frac{dx''}{dz}\cr',...
'\frac{dy''}{dx} & \frac{dy''}{dy} & \frac{dy''}{dz}\cr',...
'\frac{dz''}{dx} & \frac{dz''}{dy} & \frac{dz''}{dz}\cr',...
'\end{pmatrix}\end{eqnarray*}*/'],...
[...
'The value of dx''/dy is a measure of how much x'' changes if y is changed by a ',...
'tiny amount. ',...
'The determinant of the Jacobian is the measure of relative volumes of warped ',...
'and unwarped structures.  The modulation step simply involves multiplying by ',...
'the relative volumes /*(see Figure \ref{seg2})*/.']};

%------------------------------------------------------------------------

estwrite      = cfg_exbranch;
estwrite.tag = 'estwrite';
estwrite.name = 'VBM8: Estimate & Write';
estwrite.val = {data,opts,output,extopts};
estwrite.prog   = @cg_vbm8_run;
estwrite.help   = {[...
'This toolbox is currently only work in progress, and is an extension of the default ',...
'unified segmentation.  The algorithm is essentially the same as that described in the ',...
'Unified Segmentation paper, except for (i) a slightly different treatment of the mixing ',...
'proportions, (ii) the use of an improved registration model, ',...
'(iii) the ability to use multi-spectral data, (iv) an extended set of ',...
'tissue probability maps, which allows a different treatment of voxels outside the brain. ',...
'Some of the options in the toolbox do not yet work, and it has not yet been seamlessly integrated ',...
'into the SPM8 software.  Also, the extended tissue probability maps need further refinement. ',...
'The current versions were crudely generated (by JA) using data that was kindly provided by ',...
'Cynthia Jongen of the Imaging Sciences Institute at Utrecht, NL.'],...
'',[...
'Segment, bias correct and spatially normalise - all in the same model/* \cite{ashburner05}*/. ',...
'This function can be used for bias correcting, spatially normalising ',...
'or segmenting your data.'],...
'',...
[...
'Many investigators use tools within older versions of SPM for '...
'a technique that has become known as "optimised" voxel-based '...
'morphometry (VBM). '...
'VBM performs region-wise volumetric comparisons among populations of subjects. '...
'It requires the images to be spatially normalised, segmented into '...
'different tissue classes, and smoothed, prior to performing '...
'statistical tests/* \cite{wright_vbm,am_vbmreview,ashburner00b,john_should}*/. The "optimised" pre-processing strategy '...
'involved spatially normalising subjects'' brain images to a '...
'standard space, by matching grey matter in these images, to '...
'a grey matter reference.  The historical motivation behind this '...
'approach was to reduce the confounding effects of non-brain (e.g. scalp) '...
'structural variability on the registration. '...
'Tissue classification in older versions of SPM required the images to be registered '...
'with tissue probability maps. After registration, these '...
'maps represented the prior probability of different tissue classes '...
'being found at each location in an image.  Bayes rule can '...
'then be used to combine these priors with tissue type probabilities '...
'derived from voxel intensities, to provide the posterior probability.'],...
'',...
[...
'This procedure was inherently circular, because the '...
'registration required an initial tissue classification, and the '...
'tissue classification requires an initial registration.  This circularity '...
'is resolved here by combining both components into a single '...
'generative model. This model also includes parameters that account '...
'for image intensity non-uniformity. '...
'Estimating the model parameters (for a maximum a posteriori solution) '...
'involves alternating among classification, bias correction and registration steps. '...
'This approach provides better results than simple serial applications of each component.']};

%------------------------------------------------------------------------

write      = cfg_exbranch;
write.tag = 'write';
write.name = 'VBM8: Write already estimated segmentations';
write.val = {data,output,extopts};
write.prog   = @cg_vbm8_run;
write.help   = {[...
'Allows previously estimated segmentations (stored in imagename''_seg8.mat'' files) ',...
'to save the segmented images only without estimating the segmentation again. ',...
'This might be helpful if you have already estimated segmentations and you need ',...
'an additional tissue class, or you want to change voxel size of segmented images,']};

%------------------------------------------------------------------------
tools = cg_vbm8_tools;
%------------------------------------------------------------------------

vbm8  = cfg_choice;
vbm8.name = 'VBM8';
vbm8.tag  = 'vbm8';
%vbm8.values = {estwrite,write,tools};
vbm8.values = {estwrite,write};
%vbm8.vout = @vout;
%------------------------------------------------------------------------

