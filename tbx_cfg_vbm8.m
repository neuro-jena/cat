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

%_______________________________________________________________________
tpm = cfg_files;
tpm.tag  = 'tpm';
tpm.name = 'Tissue Probability Map';
tpm.help    = {
               'Select the tissue probability image for this class. These should be maps of eg grey matter, white matter or cerebro-spinal fluid probability. A nonlinear deformation field is estimated that best overlays the tissue probability maps on the individual subjects'' image. The default tissue probability maps are modified versions of the ICBM Tissue Probabilistic Atlases. These tissue probability maps are kindly provided by the International Consortium for Brain Mapping, John C. Mazziotta and Arthur W. Toga. http://www.loni.ucla.edu/ICBM/ICBM_TissueProb.html. The original data are derived from 452 T1-weighted scans, which were aligned with an atlas space, corrected for scan inhomogeneities, and classified into grey matter, white matter and cerebrospinal fluid. These data were then affine registered to the MNI space and down-sampled to 2mm resolution.'
               ''
               'Rather than assuming stationary prior probabilities based upon mixing proportions, additional information is used, based on other subjects'' brain images. Priors are usually generated by registering a large number of subjects together, assigning voxels to different tissue types and averaging tissue classes over subjects. The algorithm used here will employ these priors for the first initial segmentation and normalization. Six tissue classes are used: grey matter, white matter, cerebro-spinal fluid, bone, non-brain soft tissue and air outside of the head and in nose, sinus and ears. These maps give the prior probability of any voxel in a registered image being of any of the tissue classes - irrespective of its intensity.'
               ''
               'The model is refined further by allowing the tissue probability maps to be deformed according to a set of estimated parameters. This allows spatial normalisation and segmentation to be combined into the same model.'
               ''
               'Selected tissue probability map must be in multi-volume nifti format and contain all six tissue priors.'
               }';
tpm.filter = 'image';
tpm.ufilter = '.*';
tpm.def  = @(val)cg_vbm8_get_defaults('opts.tpm', val{:});
tpm.num     = [1 1];

%------------------------------------------------------------------------
% various options for estimating the segmentations
%------------------------------------------------------------------------

ngaus      = cfg_entry;
ngaus.tag  = 'ngaus';
ngaus.name = 'Gaussians per class';
ngaus.strtype = 'e';
ngaus.num = [1 6];
ngaus.def  = @(val)cg_vbm8_get_defaults('opts.ngaus', val{:});
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
biasreg.def  = @(val)cg_vbm8_get_defaults('opts.biasreg', val{:});
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
biasfwhm.def  = @(val)cg_vbm8_get_defaults('opts.biasfwhm', val{:});
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
warpreg.def  = @(val)cg_vbm8_get_defaults('opts.warpreg', val{:});
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
affreg.def  = @(val)cg_vbm8_get_defaults('opts.affreg', val{:});
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

samp      = cfg_entry;
samp.tag = 'samp';
samp.name = 'Sampling distance';
samp.strtype = 'e';
samp.num = [1 1];
samp.def  = @(val)cg_vbm8_get_defaults('opts.samp', val{:});
samp.help    = {'This encodes the approximate distance between sampled points when estimating the model parameters. Smaller values use more of the data, but the procedure is slower and needs more memory. Determining the ''''best'''' setting involves a compromise between speed and accuracy.'};

%------------------------------------------------------------------------

opts      = cfg_branch;
opts.tag = 'opts';
opts.name = 'Estimation options';
opts.val = {tpm,ngaus,biasreg,biasfwhm,affreg,warpreg,samp};
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

vox      = cfg_entry;
vox.tag = 'vox';
vox.name = 'Voxel size for normalized images';
vox.strtype = 'e';
vox.num = [1 1];
vox.def  = @(val)cg_vbm8_get_defaults('extopts.vox', val{:});
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
cleanup.labels = {'No Cleanup','Light Cleanup','Thorough Cleanup'};
cleanup.values = {0 1 2};
cleanup.def  = @(val)cg_vbm8_get_defaults('extopts.cleanup', val{:});

%------------------------------------------------------------------------

finalmask = cfg_menu;
finalmask.tag  = 'finalmask';
finalmask.name = 'Apply final mask after segmenting';
finalmask.help = {[...
'This uses a morphological operations to apply a final masking using morphological operations.']};
finalmask.labels = {'Dont apply final masking','Apply final masking'};
finalmask.values = {0 1};
finalmask.def  = @(val)cg_vbm8_get_defaults('extopts.finalmask', val{:});

%------------------------------------------------------------------------

print    = cfg_menu;
print.tag = 'print';
print.name = 'Display and print results';
print.labels = {'yes','no'};
print.values = {1 0};
print.def  = @(val)cg_vbm8_get_defaults('extopts.print', val{:});
print.help = {[...
'The normalized T1 image and the normalized segmentations can be displayed and printed to a ',...
'ps-file. This is often helpful to check whether registration and segmentation were successful. ',...
'However, this is only working if you write normalized images.']};

%------------------------------------------------------------------------

dartelwarp    = cfg_menu;
dartelwarp.tag = 'dartelwarp';
dartelwarp.name = 'Spatial normalization';
dartelwarp.labels = {'Low-dimensional: SPM default','High-dimensional: Dartel'};
dartelwarp.values = {0 1};
dartelwarp.def  = @(val)cg_vbm8_get_defaults('extopts.dartelwarp', val{:});
dartelwarp.help    = {'Choose between standard spatial normalization and high-dimensional Dartel normalization. Dartel normalized images are indicated by an additional ''''r'''' (e.g. wrp*). '};

%------------------------------------------------------------------------

extopts      = cfg_branch;
extopts.tag = 'extopts';
extopts.name = 'Extended options';
extopts.val = {dartelwarp,cleanup,print};
extopts.help = {'Extended options'};

%------------------------------------------------------------------------
% options for data
%------------------------------------------------------------------------

native    = cfg_menu;
native.tag = 'native';
native.name = 'Native space';
native.labels = {'none','yes'};
native.values = {0 1};
native.help    = {'The native space option allows you to produce a tissue class image (p*) that is in alignment with the original/* (see Figure \ref{seg1})*/. It can also be used for ''''importing'''' into a form that can be used with the DARTEL toolbox (rp*).'};

warped    = cfg_menu;
warped.tag = 'warped';
warped.name = 'Normalized';
warped.labels = {'none','yes'};
warped.values = {0 1};
warped.help = {'Write image in normalized space.'};

affine    = cfg_menu;
affine.tag = 'affine';
affine.name = 'Affine';
affine.labels = {'none','yes'};
affine.values = {0 1};
affine.help = {'Write image in normalized space, but restricted to affine transformation.'};

dartel    = cfg_menu;
dartel.tag = 'dartel';
dartel.name = 'DARTEL export';
dartel.labels = {'none','rigid (SPM8 default)','affine'};
dartel.values = {0 1 2};
dartel.help = {['This option is to export data into a form that can be used with DARTEL.',...
'The SPM8 default is to only apply rigid body transformation. An additional option is to ',...
'apply affine transformation.']};

native.def  = @(val)cg_vbm8_get_defaults('output.bias.native', val{:});
warped.def  = @(val)cg_vbm8_get_defaults('output.bias.warped', val{:});
affine.def  = @(val)cg_vbm8_get_defaults('output.bias.affine', val{:});
bias      = cfg_branch;
bias.tag = 'bias';
bias.name = 'Bias Corrected';
bias.val = {native, warped, affine};
bias.help = {[...
'This is the option to save a bias corrected version of your image. ',...
'MR images are usually corrupted by a smooth, spatially varying artifact that modulates the intensity ',...
'of the image (bias). ',...
'These artifacts, although not usually a problem for visual inspection, can impede automated ',...
'processing of the images. The bias corrected version should have more uniform intensities within ',...
'the different types of tissues and can be saved in native space and/or normalised.']};

%------------------------------------------------------------------------

warped.def  = @(val)cg_vbm8_get_defaults('output.jacobian.warped', val{:});
jacobian      = cfg_branch;
jacobian.tag = 'jacobian';
jacobian.name = 'Jacobian determinant';
jacobian.val = {warped};
jacobian.help = {[...
'This is the option to save the Jacobian determinant, which expresses local volume changes. This image can be used in a pure deformation based morphometry (DBM) design.']};

%------------------------------------------------------------------------

native.def  = @(val)cg_vbm8_get_defaults('output.label.native', val{:});
warped.def  = @(val)cg_vbm8_get_defaults('output.label.warped', val{:});
dartel.def  = @(val)cg_vbm8_get_defaults('output.label.dartel', val{:});

label      = cfg_branch;
label.tag = 'label';
label.name = 'PVE label image';
label.val = {native, warped, dartel};
label.help = {[...
'This is the option to save a labeled version of your segmentations. ',...
'Labels are saved as Partial Volume Estimation (PVE) values with different mix classes for GM-WM and GM-CSF.']};

%------------------------------------------------------------------------

modulated    = cfg_menu;
modulated.tag = 'modulated';
modulated.name = 'Modulated normalized';
modulated.labels = {'none','affine + non-linear (SPM8 default)','non-linear only'};
modulated.values = {0 1 2};
modulated.help = {[...
'``Modulation'''' is to compensate for the effect of spatial normalisation. Spatial normalisation ',...
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

native.def    = @(val)cg_vbm8_get_defaults('output.GM.native', val{:});
warped.def    = @(val)cg_vbm8_get_defaults('output.GM.warped', val{:});
modulated.def = @(val)cg_vbm8_get_defaults('output.GM.mod', val{:});
dartel.def    = @(val)cg_vbm8_get_defaults('output.GM.dartel', val{:});
grey      = cfg_branch;
grey.tag = 'GM';
grey.name = 'Grey matter';
grey.val = {native, warped, modulated, dartel};
grey.help     = {'Options to produce grey matter images: p1*.img, wp1*.img and mwp1*.img.'};

native.def    = @(val)cg_vbm8_get_defaults('output.WM.native', val{:});
warped.def    = @(val)cg_vbm8_get_defaults('output.WM.warped', val{:});
modulated.def = @(val)cg_vbm8_get_defaults('output.WM.mod', val{:});
dartel.def    = @(val)cg_vbm8_get_defaults('output.WM.dartel', val{:});
white      = cfg_branch;
white.tag = 'WM';
white.name = 'White matter';
white.val = {native, warped, modulated, dartel};
white.help    = {'Options to produce white matter images: p2*.img, wp2*.img and mwp2*.img.'};

native.def    = @(val)cg_vbm8_get_defaults('output.CSF.native', val{:});
warped.def    = @(val)cg_vbm8_get_defaults('output.CSF.warped', val{:});
modulated.def = @(val)cg_vbm8_get_defaults('output.CSF.mod', val{:});
dartel.def    = @(val)cg_vbm8_get_defaults('output.CSF.dartel', val{:});
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
    'none',...
    'Image->Template (forward)',...
    'Template->Image (inverse)',...
    'inverse + forward'};
warps.values = {[0 0],[1 0],[0 1],[1 1]};
warps.def  = @(val)cg_vbm8_get_defaults('output.warps', val{:});
warps.help    = {'Deformation fields can be saved to disk, and used by the Deformations Utility. For spatially normalising images to MNI space, you will need the forward deformation, whereas for spatially normalising (eg) GIFTI surface files, you''ll need the inverse. It is also possible to transform data in MNI space on to the individual subject, which also requires the inverse transform. Deformations are saved as .nii files, which contain three volumes to encode the x, y and z coordinates.'};

%------------------------------------------------------------------------

output      = cfg_branch;
output.tag = 'output';
output.name = 'Writing options';
output.val = {grey, white, csf, bias, label, jacobian, warps};
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
'the original/* (see Figure \ref{seg1})*/. You can also produce spatially normalised versions - both ',...
'with (mwc*) and without (wc*) modulation/* (see Figure \ref{seg2})*/. In the vbm8 toolbox, the voxel size ',...
'of the spatially normalised versions is 1.5 x 1.5 x 1.5mm as default. The HMRF (hidden Markov Random Fields), ',...
'which were optional in the vbm5 toolbox are now calculated by defaults. Their weighting is estimated ',...
'automatically in dependence of the noise in the data. ',...
'The produced images of the tissue classes can directly be used for doing voxel-based morphometry (both un-modulated and modulated). ',...
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
estwrite.val = {data,opts,extopts,output};
estwrite.prog   = @cg_vbm8_run;
estwrite.vout = @vout;
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
write.val = {data,extopts,output};
write.prog   = @cg_vbm8_run;
write.vout = @vout;
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
vbm8.values = {estwrite,write,tools};
%------------------------------------------------------------------------

%------------------------------------------------------------------------
function dep = vout(job)

opts  = job.output;
tissue(1).warped = [opts.GM.warped  (opts.GM.modulated==1)  (opts.GM.modulated==2) ];
tissue(1).native = [opts.GM.native  (opts.GM.dartel==1)     (opts.GM.dartel==2)    ];
tissue(2).warped = [opts.WM.warped  (opts.WM.modulated==1)  (opts.WM.modulated==2) ];
tissue(2).native = [opts.WM.native  (opts.WM.dartel==1)     (opts.WM.dartel==2)    ];
tissue(3).warped = [opts.CSF.warped (opts.CSF.modulated==1) (opts.CSF.modulated==2)];
tissue(3).native = [opts.CSF.native (opts.CSF.dartel==1)    (opts.CSF.dartel==2)   ];

% This depends on job contents, which may not be present when virtual
% outputs are calculated.

cdep = cfg_dep;
cdep(end).sname      = 'Seg Params';
cdep(end).src_output = substruct('.','param','()',{':'});
cdep(end).tgt_spec   = cfg_findspec({{'filter','mat','strtype','e'}});
if opts.bias.native,
    cdep(end+1)          = cfg_dep;
    cdep(end).sname      = 'Bias Corr Images';
    cdep(end).src_output = substruct('()',{1}, '.','biascorr','()',{':'});
    cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
end;
if opts.bias.warped,
    cdep(end+1)          = cfg_dep;
    cdep(end).sname      = 'Warped Bias Corr Images';
    cdep(end).src_output = substruct('()',{1}, '.','wbiascorr','()',{':'});
    cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
end;
if opts.label.native,
    cdep(end+1)          = cfg_dep;
    cdep(end).sname      = 'Label Images';
    cdep(end).src_output = substruct('()',{1}, '.','label','()',{':'});
    cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
end;
if opts.label.warped,
    cdep(end+1)          = cfg_dep;
    cdep(end).sname      = 'Warped Label Images';
    cdep(end).src_output = substruct('()',{1}, '.','wlabel','()',{':'});
    cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
end;
if opts.label.dartel==1,
    cdep(end+1)          = cfg_dep;
    cdep(end).sname      = 'Rigid Registered Label Images';
    cdep(end).src_output = substruct('()',{1}, '.','rlabel','()',{':'});
    cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
end;
if opts.label.dartel==2,
    cdep(end+1)          = cfg_dep;
    cdep(end).sname      = 'Affine Registered Label Images';
    cdep(end).src_output = substruct('()',{1}, '.','alabel','()',{':'});
    cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
end;
if opts.jacobian.warped,
    cdep(end+1)          = cfg_dep;
    cdep(end).sname      = 'Jacobian Determinant Images';
    cdep(end).src_output = substruct('()',{1}, '.','jacobian','()',{':'});
    cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
end;
if opts.warps(1),
    cdep(end+1)          = cfg_dep;
    cdep(end).sname      = 'Deformation Field';
    cdep(end).src_output = substruct('()',{1}, '.','fordef','()',{':'});
    cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
end;
if opts.warps(2),
    cdep(end+1)          = cfg_dep;
    cdep(end).sname      = 'Inverse Deformation Field';
    cdep(end).src_output = substruct('()',{1}, '.','invdef','()',{':'});
    cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
end;

for i=1:numel(tissue),
    if tissue(i).native(1),
        cdep(end+1)          = cfg_dep;
        cdep(end).sname      = sprintf('p%d Images',i);
        cdep(end).src_output = substruct('.','tiss','()',{i},'.','c','()',{':'});
        cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
    end
    if tissue(i).native(2),
        cdep(end+1)          = cfg_dep;
        cdep(end).sname      = sprintf('rp%d rigid Images',i);
        cdep(end).src_output = substruct('.','tiss','()',{i},'.','rc','()',{':'});
        cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
    end
    if tissue(i).native(3),
        cdep(end+1)          = cfg_dep;
        cdep(end).sname      = sprintf('rp%d affine Images',i);
        cdep(end).src_output = substruct('.','tiss','()',{i},'.','rca','()',{':'});
        cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
    end
    if tissue(i).warped(1),
        cdep(end+1)          = cfg_dep;
        cdep(end).sname      = sprintf('wp%d Images',i);
        cdep(end).src_output = substruct('.','tiss','()',{i},'.','wc','()',{':'});
        cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
    end
    if tissue(i).warped(2),
        cdep(end+1)          = cfg_dep;
        cdep(end).sname      = sprintf('mwp%d Images',i);
        cdep(end).src_output = substruct('.','tiss','()',{i},'.','mwc','()',{':'});
        cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
    end
    if tissue(i).warped(3),
        cdep(end+1)          = cfg_dep;
        cdep(end).sname      = sprintf('m0wp%d Images',i);
        cdep(end).src_output = substruct('.','tiss','()',{i},'.','m0wc','()',{':'});
        cdep(end).tgt_spec   = cfg_findspec({{'filter','image','strtype','e'}});
    end
end

dep = cdep;



