load sampleData;

paramsData.weightsPerSequence = ones(1,128) ;
paramsData.factorSeqWeights = 1;

%Simulation of real-time CRF. Real-time LDCRF is extactly the same, just replace
%CRF by LDCRF. There is no function to insert frames from MATLAB yet, so
%RTCRF support for MATLAB as of now is just to simulate them.

%Delay that we plan to use for the real-time CRF.
%Basically, means 'delay' observations will be kept in a buffer, and
%inference will be done on this buffer using forward-backward (alpha-beta)
%propagation.
params.delay = 1;
params.caption = 'RTCRF';
params.modelType = 'crfRealtime';
%For LDCRF, replace the modelType by 'ldcrfRealtime' and uncomment the next
%line
%params.nbHiddenStates = 2;
params.optimizer = 'bfgs';
params.regFactor = 10;
params.windowSize = 0;
params.nbIterations = 1;
params.maxIterations = 300;
params.validateParams = {'regFactor'};
params.validateValues = {[0.1 1 10]};
params.debugLevel = 0;
params.computeTrainingError = 1;
params.rangeWeights = [-1 1];
params.normalizeWeights = 1;

%Function train reads params.modelType to determine which model to train.
[model stats] = train(trainSeqs, trainLabels, params);

%Test for different values of delay.
delays = 1:20;
R = testWithSameModel(testSeqs,testLabels,params,model,delays);

plotMarginalErrors(R{end},{R{1:end-1}});


