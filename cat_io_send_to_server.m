function cat_io_send_to_server(urlinfo)
% ______________________________________________________________________
% 
% Send status information to Piwik server
%
%   cat_io_send_to_server(urlinfo);
%
%   urlinfo      .. piwik status information
% ______________________________________________________________________
%
%   Christian Gaser (christian.gaser@uni-jena.de)
%   Structural Brain Mapping Group (http://www.neuro.uni-jena.de/)
%   Department of Neurology
%   University Jena
% ______________________________________________________________________
% $Id$
  
urlinfo = regexprep(urlinfo, '\n', '%20'); % replace returns
urlinfo = regexprep(urlinfo, ' ' , '%20'); % replace spaces

piwikserver = 'http://141.35.69.218/piwik/piwik.php?idsite=1&rec=1&action_name=';
url = sprintf('%s%s',piwikserver,urlinfo);

try
  [s,sts] = urlread(url,'Timeout',2);
catch
  [s,sts] = urlread(url);
end
