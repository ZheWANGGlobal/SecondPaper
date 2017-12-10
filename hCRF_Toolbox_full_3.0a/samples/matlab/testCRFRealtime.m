% [ll newlabels] = testCRFRealtime(model, seqs, labels)
%     Test sequences using a Realtime CRF model: Compute the log likelihood for each
%     sequence. Also, based on the labels, compute error measurements.
%
%     This function will restrict the testing to view only 'delay' frames
%     ahead of the current frame.
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
function [ll newLabels] = testCRFRealtime(modelCRF_RT, seqs, labels)

intLabels = cellInt32(labels);
matHCRF('createToolbox','crf');
if isfield(modelCRF_RT,'windowSize')
    matHCRF('addFeatureFunction','StartFeatures');
    matHCRF('addFeatureFunction','BackwardWindowRawFeatures',modelCRF_RT.windowSize);
end

%This inference engine and the different feature functions are what makes a
%real-time CRF.
matHCRF('setInferenceEngine','FF',modelCRF_RT.delay);
matHCRF('initToolbox');

matHCRF('setData',seqs,intLabels);
matHCRF('setModel',modelCRF_RT.model, modelCRF_RT.features);
if isfield(modelCRF_RT,'debugLevel')
    matHCRF('set','debugLevel',modelCRF_RT.debugLevel);
end

matHCRF('test');
    
ll=matHCRF('getResults');
newLabels = labels;