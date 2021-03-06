% [ll newlabels] = testLDCRFRealtime(model, seqs, labels)
%   Simulate real-time testing by inserting frames one by one in a DataSequenceRealtime object.  
%
%   Test sequences using a Latent-Dynamic CRF model: Compute the log
%     likelihood for each sequence. Also, based on the labels, compute
%     error measurements. 
% 
% INPUT:
%    - model            : Model parameters as returned by the Train
%                         function.
%    - seqs             : Cell array of matrices which contains the encoded
%                         features for each sequence
%    - labels           : Cell array of vectors which contains the ground
%                         truth label for each sample.
%    OUTPUT:
%    - ll               : Log likelyhood of a label for each sample. For
%                         some models, equal to the maginal probability.
%    - newLabels        : Ground truth label for each sample. In most
%                         cases, same as labels. For HMM and HCRF, the
%                         number of samples may be slightly different.
function [ll newLabels] = testLDCRFRealtime(modelLDCRF, seqs, labels)

intLabels = cellInt32(labels);
matHCRF('createToolbox','ldcrf',modelLDCRF.nbHiddenStates);
if isfield(modelLDCRF,'windowSize')
    matHCRF('addFeatureFunction','StartFeatures');
    matHCRF('addFeatureFunction','BackwardWindowRawFeatures',modelLDCRF.windowSize);
end
%This inference engine and the different feature functions are what makes a
%real-time LDCRF. Only set it for testing, because training has issues.
matHCRF('setInferenceEngine','FF',modelLDCRF.delay);
%Explicit call to init toolbox required for testing.
matHCRF('initToolbox');

matHCRF('setData',seqs,intLabels);
matHCRF('setModel',modelLDCRF.model, modelLDCRF.features);
if isfield(modelLDCRF,'debugLevel')
    matHCRF('set','debugLevel',modelLDCRF.debugLevel);
end

matHCRF('test');

ll=matHCRF('getResults');
newLabels = labels;