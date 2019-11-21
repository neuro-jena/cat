%CAT_VOL_GENUS0 Topology adjusted isosurface extractor.
%  Toplogy optimized marching cubes surface creation for an existing volume 
%  Y and a treshold th.  Also outputs the topology adjusted volume Yc.
%
%  [Yc,faces,vertices] = cat_vol_genus0(Y,th[,notadjust]);
%  
%  Yc         .. topolocy adjusted volume 
%  faces      .. surface faces
%  vertices   .. surface vertices
%  Y          .. input volume as single datatype
%  th         .. threshold used for input volume Y
%  notadjust  .. no topology adjustment
%
%  To avoid output use the MATLAB EVALC function. 
%
%  See also ISOSURFACE, EVALC, compile.
%  ________________________________________________________________________
%  $Id: cat_conf_stools.m 1519 2019-11-19 10:48:29Z gaser $
