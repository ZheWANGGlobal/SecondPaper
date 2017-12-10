% [modelLDCNF stats] = trainLDCNF(seqs, labels, params)
%     Train LDCNF model based on feature sequences and corresponding labels.
% 
% INPUT:
%    - seqs             : Cell array of matrices which contains the encoded
%                         features for each sequence
%    - labels           : Cell array of vectors which contains the ground
%                         truth label for each sample.
%    - params           : Parameters for the training procedure.
%
%    OUTPUT:
%    - modelLDCNF       : internal paramters from the trained model
%    - stats            : Statistic from the training procedure (e.g.,
%                         gradient norm, likelyhood error, training time)
function [modelLDCNF stats] = trainLDCNF(seqs, labels, params) 

intLabels = cellInt32(labels);

matHCRF('createToolbox','ldcrf', params.nbHiddenStates);

if isfield(params,'optimizer')
    matHCRF('setOptimizer',params.optimizer)
end

if isfield(params,'windowSize')
    matHCRF('addFeatureFunction','StartFeatures');
    matHCRF('addFeatureFunction','GateNodeFeatures',params.nbGates,params.windowSize);
end

matHCRF('setGradient','GradientLDCNF');

matHCRF('setData',seqs,intLabels);
if isfield(params,'rangeWeights')
    matHCRF('set','minRangeWeights',params.rangeWeights(1));
    matHCRF('set','maxRangeWeights',params.rangeWeights(2));
end
if isfield(params,'weightsInitType')
    matHCRF('set','weightsInitType',params.weightsInitType);
end    
if isfield(params,'debugLevel')
    matHCRF('set','debugLevel',params.debugLevel);
end
if isfield(params,'regFactor') % For backward compatibility
    matHCRF('set','regularizationL2',params.regFactor);
end
if isfield(params,'regFactorL2')
    matHCRF('set','regularizationL2',params.regFactorL2);
end
if isfield(params,'regFactorL1')
    matHCRF('set','regularizationL1',params.regFactorL1);
end
if isfield(params,'regL1FeatureTypes')
    matHCRF('set','regL1FeatureTypes',params.regL1FeatureTypes);
end
if isfield(params,'maxIterations')
    matHCRF('set','maxIterations',params.maxIterations);
end
if isfield(params,'initWeights')
    matHCRF('set','initWeights',params.initWeights);
end
if isfield(params,'nbThreads')
    matHCRF('set','nbThreads',params.nbThreads);
end

% RESET RND GENERATOR TO DESIRED SEED 
if isfield(params, 'useSameSeedPerIterationNb') && params.useSameSeedPerIterationNb %{+KGB}
    assert(isfield(params, 'seeds'), 'modelParams.seeds were not set, but useSameSeedPerIterationNb switch was on'); %{+KGB}
    assert(isfield(params, 'seedIndex'), 'seedIndex not set'); %{+KGB}        
    matHCRF('set', 'randomSeed', params.seeds(params.seedIndex)); %{+KGB}    
end

matHCRF('train');
[modelLDCNF.model modelLDCNF.features] = matHCRF('getModel');
modelLDCNF.optimizer = params.optimizer;
modelLDCNF.nbHiddenStates = params.nbHiddenStates;
modelLDCNF.nbGates = params.nbGates;
modelLDCNF.regFactor = params.regFactor;
modelLDCNF.windowSize = params.windowSize;
modelLDCNF.debugLevel = params.debugLevel;
modelLDCNF.modelType = params.modelType;
modelLDCNF.caption = params.caption;

stats.NbIterations = matHCRF('get','statsNbIterations');
stats.FunctionError = matHCRF('get','statsFunctionError');
stats.NormGradient = matHCRF('get','statsNormGradient');
