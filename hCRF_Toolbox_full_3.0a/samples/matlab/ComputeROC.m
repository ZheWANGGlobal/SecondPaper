% function [truePositives falsePositives] = ComputeROC(ll,labels, modelType, rocRange)
%   computes the true positive and false positives rates for different
%   thresholds.
%
% INPUT:
%   - ll                : cell with the probabilities result of the learned
%                         model for each sequence
%   - labels            : cell with the label encoding for the real gestures 
%                         you are trying to predict
%   - modelType         : the type of model which created the ll
%                         probabilities
%   - rocRange          : (optional) the number of thresholds you want to
%                         use (1000 by default)
%
% OUTPUT:
%   - truePositives     : the true positive rates for each threshold
%   - falseNegatives    : the false 
%
% Documentation added by Iwan de Kok 4/15/2008
%
% Modified by Yale Song 05/16/2011 to add support for HCRF family

function [truePositives falsePositives] = ComputeROC(ll,labels, modelType, rocRange, labelid)

if ~exist('rocRange','var')
    rocRange = 1000;
end
if ~exist('labelid','var')
    labelid = 1;
end

matLabels = cell2mat(labels);
matLikelihoods = cell2mat(ll);

if size(matLikelihoods,1) == 1
    [truePositives falsePositives] = CreateROC(matLabels,matLikelihoods(1,:),rocRange);
elseif exist('modelType','var') && (~isempty(strfind(modelType,'hcrf')) || strcmp(modelType,'hmm'))
    [truePositives falsePositives] = CreateROC(matLabels,matLikelihoods(2,:)-matLikelihoods(1,:),rocRange);
else
    [truePositives falsePositives] = CreateROC(matLabels,matLikelihoods(labelid+1,:),rocRange);
end
