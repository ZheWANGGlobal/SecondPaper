% [ll newlabels] = testLDCNF(model, seqs, labels)
%     Test sequences using a Latent-Dynamic CNF model: Compute the log
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
function [ll newLabels] = testLDCNF(modelLDCNF, seqs, labels)

intLabels = cellInt32(labels);
matHCRF('createToolbox','ldcrf', modelLDCNF.nbHiddenStates);

if isfield(modelLDCNF,'windowSize')
    matHCRF('addFeatureFunction','StartFeatures');
    matHCRF('addFeatureFunction','GateNodeFeatures',modelLDCNF.nbGates, modelLDCNF.windowSize);
end

matHCRF('initToolbox');

matHCRF('setData',seqs,intLabels);
matHCRF('setModel',modelLDCNF.model, modelLDCNF.features);
if isfield(modelLDCNF,'debugLevel')
    matHCRF('set','debugLevel',modelLDCNF.debugLevel);
end
matHCRF('test');

ll=matHCRF('getResults');
newLabels = labels;