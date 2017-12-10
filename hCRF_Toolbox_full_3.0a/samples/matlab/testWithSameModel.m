function [Rc] = testWithSameModel(seqs, labels, params, model, testedParams)
%function [Rc] = testWithSameModel(seqs, labels, params, model, testedParams)
%TESTWITHSAMEMODEL 

if params.windowSize ~= 0
    error('Cannot test both CRF and CRF-Realtime for comparison in one function if the windowSize is non-null. Make your own script or try with windowSize 0 :)');
end

if ~isfield(params,'errorType')
    params.errorType = 'eer';
end
if ~isfield(params,'rocRange')
    params.rocRange = 1000;
end
if ~isfield(params,'allErrors')
    params.allErrors = 1;
end

Rc = {};
for j=1:size(testedParams,2)
    clear R;
    %Hardcoded for delay, I know, terrible.
    params.delay = testedParams(j);
    model.delay = testedParams(j);
    R.params = params;
       
    [R.ll R.labels] = test(model, seqs, labels);
    [R.stats.trueRates R.stats.falseRates] = ComputeROC(R.ll,R.labels,params.modelType,params.rocRange);
    R.testError = computeEqualRate(R.stats.trueRates,R.stats.falseRates);
    
    Rc{j} = R;
end

%Finally, compare with base.
params.delay = 0;
model.delay = 0;
clear R;
if strcmp(params.modelType,'ldcrfRealtime')
    params.modelType = 'ldcrf';
    model.modelType = 'ldcrf';
else
    params.modelType = 'crf';
    model.modelType = 'crf';
end

[R.ll R.labels] = test(model, seqs, labels);
[R.stats.trueRates R.stats.falseRates] = ComputeROC(R.ll,R.labels,params.modelType,params.rocRange);
R.testError = computeEqualRate(R.stats.trueRates,R.stats.falseRates);

Rc{end+1} = R;


end

