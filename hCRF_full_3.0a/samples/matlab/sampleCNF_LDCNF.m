load sampleData;

%Sample script training a CNF and LDCNF and testing them on the sample
%dataset. For more information, look at the file trainCNF and testCNF or
%trainLDCNF and testLDCNF.

%Be aware that the training of a CNF is non convex, and even more for
%LDCNFs. This means that you should always train them a few times and pick
%the best one. This is not done in this sample to keep things simple.

R = {};

%CNF, Conditional neural fields. Add a single layer neural network with
%nbGates gates to preprocess the input features, thus extracting non-linear
%relationships between features that a regular crf would miss.
R{1}.params.caption = 'CNF';
R{1}.params.modelType = 'cnf';
R{1}.params.optimizer = 'lbfgs';
R{1}.params.regFactor = 1;
%This is the number of gates that will be used to preprocess the input
%features. Do a small grid-search to find the best number of gates.
R{1}.params.nbGates = 5;
R{1}.params.windowSize = 0;
R{1}.params.nbIterations = 1;
R{1}.params.maxIterations = 100;
R{1}.params.debugLevel = 0;
R{1}.params.computeTrainingError = 1;
R{1}.params.rocRange = 1000;
R{1}.params.rangeWeights = [-1 1];
R{1}.params.normalizeWeights = 1;

%Function train reads params.modelType to determine which model to train.
[R{1}.model R{1}.stats] = train(trainSeqs, trainLabels, R{1}.params);
[R{1}.likelihoods R{1}.labels] = test(R{1}.model, testSeqs, testLabels);
[R{1}.stats.trueRates R{1}.stats.falseRates] = ComputeROC(R{1}.likelihoods,R{1}.labels,R{1}.params.modelType,R{1}.params.rocRange);
R{1}.testError = computeEqualRate(R{1}.stats.trueRates,R{1}.stats.falseRates);

%LDCNF, Latent-dynamic conditional neural fields. Same concept than CNF,
%except we preprocess the input of an LDCRF instead of a CRF.
R{2}.params.caption = 'LDCNF';
R{2}.params.modelType = 'ldcnf';
R{2}.params.optimizer = 'lbfgs';
R{2}.params.regFactor = 1;
%This is the number of gates that will be used to preprocess the input
%features. Do a small grid-search to find the best number of gates.
R{2}.params.nbGates = 5;
R{2}.params.nbHiddenStates = 2;
R{2}.params.windowSize = 0;
R{2}.params.nbIterations = 1;
R{2}.params.maxIterations = 100;
R{2}.params.debugLevel = 0;
R{2}.params.computeTrainingError = 1;
R{2}.params.rocRange = 1000;
R{2}.params.rangeWeights = [-1 1];
R{2}.params.normalizeWeights = 1;

%Function train reads params.modelType to determine which model to train.
[R{2}.model R{2}.stats] = train(trainSeqs, trainLabels, R{2}.params);
[R{2}.likelihoods R{2}.labels] = test(R{2}.model, testSeqs, testLabels);
[R{2}.stats.trueRates R{2}.stats.falseRates] = ComputeROC(R{2}.likelihoods,R{2}.labels,R{2}.params.modelType,R{2}.params.rocRange);
R{2}.testError = computeEqualRate(R{2}.stats.trueRates,R{2}.stats.falseRates);

disp(['CNF test error ' num2str(R{1}.testError)])
disp(['LDCNF test error ' num2str(R{2}.testError)])

