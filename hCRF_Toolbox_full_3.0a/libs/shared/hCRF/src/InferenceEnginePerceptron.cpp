#include "inferenceengine.h"
#include <assert.h>
#include <vector>

using namespace std;


InferenceEnginePerceptron::InferenceEnginePerceptron()
{}

InferenceEnginePerceptron::~InferenceEnginePerceptron()
{}

void InferenceEnginePerceptron::computeMi(FeatureGenerator* fGen, Model* model, DataSequence* X,
                      int i, int seqLabel, dMatrix& Mi_YY, dVector& Ri_Y,
                      bool bUseStatePerNodes)
// This function compute Ri_Y = Sum of features for every states at time i
	// and Mi_YY, the sum of feature for every transition (i-1, i). 
{	
	Mi_YY.set(0);
	Ri_Y.set(0);
	dVector* lambda = model->getWeights(seqLabel);
	
	// Compute Ri_Y
#if defined(_VEC_FEATURES) || defined(_OPENMP)
	featureVector vecFeature;
	fGen->getFeatures(vecFeature, X,model,i,-1,seqLabel);
	feature* pFeature = vecFeature.getPtr();
	for (int j = 0; j < vecFeature.size() ; j++, pFeature++)
#else
	featureVector* vecFeature = fGen->getFeatures(X,model, i, -1 ,seqLabel);
	feature* pFeature = vecFeature->getPtr();
	for (int j = 0; j < vecFeature->size() ; j++, pFeature++)
#endif
	{
		int f = pFeature->id;
		int yp = pFeature->nodeState;
		double val = pFeature->value;
		double oldVal = Ri_Y.getValue(yp);
		Ri_Y.setValue(yp,oldVal+(*lambda)[f]*val);
	}	
	// Compute Mi_YY
	if(i>0) {
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		fGen->getFeatures(vecFeature, X, model, i, i-1, seqLabel);
		pFeature = vecFeature.getPtr();
		for (int j = 0; j < vecFeature.size() ; j++, pFeature++) {
#else
		vecFeature = fGen->getFeatures(X,model,i,i-1,seqLabel);
		pFeature = vecFeature->getPtr();
		for (int j = 0; j < vecFeature->size() ; j++, pFeature++) {
#endif
			int f = pFeature->id;
			int yp = pFeature->nodeState;
			int yprev = pFeature->prevNodeState;
			double val = pFeature->value;
			Mi_YY.setValue(yprev, yp, Mi_YY.getValue(yprev,yp)+(*lambda)[f]*val);
		}
	}
	double maskValue = -INF_VALUE;	
	
	if(bUseStatePerNodes) { 
		iMatrix* pStatesPerNodes = model->getStateMatrix(X);
		for(int s = 0; s < Ri_Y.getLength(); s++) {
			if(pStatesPerNodes->getValue(s,i) == 0)
				Ri_Y.setValue(s,maskValue);
		}
	}
}

void InferenceEnginePerceptron::ViterbiForwardMax(dMatrix& Mi_YY, dVector& Ri_Y, dVector& alpha_Y, 
					vector<iVector>& viterbiBacktrace, int nodeIndex)
{
	int NSTATES = alpha_Y.getHeight();
	dVector tmp_Y(NSTATES);
	tmp_Y.set(alpha_Y);

	for(int i=0; i<NSTATES; i++)
	{
		double max = -INF_VALUE;
		int idTraceback = -1;
		for(int j=0; j<NSTATES; j++)
		{
			double score = tmp_Y[j] + Mi_YY.getValue(j,i); 
			if(score > max)
			{
				max = score;
				idTraceback = j;
			}
		}
		alpha_Y[i] = max + Ri_Y[i];
		viterbiBacktrace[nodeIndex-1][i] = idTraceback;
	}	
}


void InferenceEnginePerceptron::computeViterbiPath(iVector& viterbiPath, FeatureGenerator* fGen,
                       DataSequence* X, Model* model,
                       int seqLabel, bool bUseStatePerNodes)
{
	if(model->getAdjacencyMatType()!=CHAIN){
		throw HcrfBadModel("InferenceEngineFB need a model based on a Chain");
	}
	
	int NNODES=X->length();	
	int NSTATES = model->getNumberOfStates();	
	
	viterbiPath.create(NNODES);	
	vector<iVector> viterbiBacktrace;
	viterbiBacktrace.resize(NNODES-1);
	for(int i =0; i < NNODES-1; i++)
	{
		viterbiBacktrace[i].create(NSTATES);
	}

	dMatrix Mi_YY (NSTATES,NSTATES);
	dVector Ri_Y (NSTATES);	
	dVector alpha_Y(NSTATES);
	
	// Compute Alpha values	
	computeMi(fGen, model, X, 0, seqLabel, Mi_YY, Ri_Y, bUseStatePerNodes);
	alpha_Y.set(Ri_Y);
	for (int i = 1; i < NNODES; i++) {
		// compute the Mi matrix		
		computeMi(fGen,model, X, i, seqLabel, Mi_YY, Ri_Y, bUseStatePerNodes);
		ViterbiForwardMax(Mi_YY, Ri_Y, alpha_Y,viterbiBacktrace,i);							
	}

	double maxValue = -INF_VALUE;
	int maxIndex = -1;
	for(int i =0; i<NNODES; i++)
	{
		double value = alpha_Y[i];
		if(value > maxIndex)
		{
			maxValue = value;
			maxIndex = i;
		}		
	}
	
	viterbiPath[NNODES-1] = maxIndex;
	int nextIndex = maxIndex;
	for(int i=NNODES-2; i>=0; i--)
	{
		viterbiPath[i] = viterbiBacktrace[i][nextIndex];
		nextIndex = viterbiPath[i];
	}
}
