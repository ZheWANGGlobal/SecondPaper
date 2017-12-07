/** 
Inference engine to be used with SharedLDCRF
Hugues Salamin, 26 jan 2010
**/

#include "inferenceengine.h"
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif

double InferenceEngineDC::computePartition(FeatureGenerator* fGen,
										   DataSequence* X, Model* model,
										   int seqLabel,
										   bool bUseStatePerNodes,bool bMaxProduct)
{
	if( bMaxProduct )
		throw HcrfNotImplemented("InferenceEngineDC for max-margin is not implemented");

	Beliefs bel;
	computeBeliefsLog(bel, fGen,X, model, true,seqLabel, bUseStatePerNodes);
	return bel.partition;
}


int InferenceEngineDC::computeSingleBelief(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob)
{
	return 0;
}


// Inference Functions
void InferenceEngineDC::computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,
									  DataSequence* X, Model* model,
									  int bComputePartition,
									  int seqLabel, bool bUseStatePerNodes,bool bMaxProduct)
{
	if( bMaxProduct )
		throw HcrfNotImplemented("InferenceEngineDC for max-margin is not implemented");

	computeBeliefsLog(bel, fGen, X, model, bComputePartition, seqLabel, bUseStatePerNodes);
}

void InferenceEngineDC::computeBeliefsLinear(Beliefs& bel, FeatureGenerator* fGen,
									  DataSequence* X, Model* model,
									  int, int, bool bUseStatePerNodes)
{
	if(model->getAdjacencyMatType()!=DANGLING_CHAIN){
		throw HcrfBadModel("InferenceEngineDC need a model based on"
						   " a DanglingChain");
	}
	int NNODES = X->length();
	int NSTATES = model->getNumberOfStates();
	int NLABELS = model->getNumberOfStateLabels();
	bel.belStates.resize(2*NNODES);
	// We have as many edges as nodes (The edge NNODES-2 does not exists, as it
	// would connect the last hidden states to the next, but the edge NNODES-1
	// exists.
	bel.belEdges.resize(2*NNODES);
	for(int i=0;i<NNODES;i++) {
		// We initialize the odd nodes that correspond to labels and the even
		// nodes that correspond to hidden states.
		bel.belStates[i].create(NLABELS);
		bel.belEdges[i+NNODES].create(NSTATES, NLABELS);
		bel.belStates[i].create(NSTATES);
		bel.belEdges[i].create(NSTATES, NSTATES);
	}
	dMatrix Mi_HH(NSTATES, NSTATES);
	dVector Ri_H (NSTATES);
	dVector Ri_Y;
	dVector Pi_Y;
	dMatrix Mi_YH;
	dVector alpha_H(NSTATES);
	dVector newAlpha_H(NSTATES);
	dVector tmp_H(NSTATES);
	bel.belStates[NNODES-1].set(1.0);
	for (int i = NNODES-1; i > 0; i--) {
		// compute the Mi matrix
		computeLogMi(fGen, model, X, i, -1, Mi_HH, Ri_H, true, 
					 bUseStatePerNodes);
		//Pi_Y is the potential on Yi, while Ri_Y is the msg from Y_i to R_i
		computeObsMsg(fGen, model, X, i, Mi_YH, Ri_Y, Pi_Y, true, 
					  bUseStatePerNodes);
		tmp_H.set(bel.belStates[i]);
		tmp_H.eltMpy(Ri_Y);
		tmp_H.eltMpy(Ri_H);
		bel.belStates[i-1].multiply(Mi_HH, tmp_H);
	}
	for (int i = 0; i < NNODES; i++) {
		// compute the Mi matrix
		computeLogMi(fGen, model, X, i, -1, Mi_HH, Ri_H, true, 
					 bUseStatePerNodes);
		computeObsMsg(fGen, model, X, i, Mi_YH, Ri_Y, Pi_Y, true, 
					  bUseStatePerNodes);
		// newAlpha_H = Ri_H + messages from H_i-1
		if (i > 0) {
			Mi_HH.transpose();
			newAlpha_H.multiply(Mi_HH, alpha_H);
			newAlpha_H.eltMpy(Ri_H);
		}
		else {
			newAlpha_H.set(Ri_H);
		}
		// We now update the beliefs on the edges between H and Y and the
		// beliefs on the Y
		Mi_YH.transpose();
		tmp_H.set(bel.belStates[i]);
		tmp_H.eltMpy(newAlpha_H);
		bel.belStates[i+NNODES].multiply(Mi_YH, tmp_H);
		tmp_H.transpose();
		bel.belEdges[i+NNODES].multiply(Pi_Y, tmp_H);
		bel.belEdges[i+NNODES].eltMpy(Mi_YH);
		// We can also compute the beliefs on the edges between the H's
		if (i > 0) {			
			tmp_H.set(Ri_H);
			tmp_H.eltMpy(bel.belStates[i]);
			tmp_H.eltMpy(Ri_Y);
			tmp_H.transpose();
			bel.belEdges[i-1].multiply(alpha_H, tmp_H);
			Mi_HH.transpose();
			bel.belEdges[i-1].eltMpy(Mi_HH);
		}
		// We update bel.belStates[i] to the sum of all the messages.
		newAlpha_H.eltMpy(Ri_Y);
		bel.belStates[i].eltMpy(newAlpha_H);
		alpha_H.set(newAlpha_H);
	}
	// We now normalize the beliefs and return the partition
	double Zx = alpha_H.sum();
	for (int i = 0; i < 2*NNODES; i++) {
		bel.belStates[i].multiply(1.0/Zx);
		bel.belEdges[i].multiply(1.0/Zx);
	}
	bel.partition = log(Zx);
}

void InferenceEngineDC::computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen, 
										  DataSequence* X,Model* model,
										  int, int, bool bUseStatePerNodes)
{	
	if(model->getAdjacencyMatType()!=DANGLING_CHAIN){
		throw HcrfBadModel("InferenceEngineDC need a model based on"
						   " a DanglingChain");
	}
	int NNODES = X->length();
	int NSTATES = model->getNumberOfStates();
	int NLABELS = model->getNumberOfStateLabels();
	bel.belStates.resize(2*NNODES);
	// We have as many edges as nodes (The edge NNODES-2 does not exists, as it
	// would connect the last hidden states to the next, but the edge NNODES-1
	// exists.
	bel.belEdges.resize(2*NNODES);
	for(int i=0;i<NNODES;i++) {
		// We initialize the odd nodes that correspond to labels and the even
		// nodes that correspond to hidden states.
		bel.belStates[i+NNODES].create(NLABELS);
		bel.belEdges[i+NNODES].create(NSTATES, NLABELS);
		bel.belStates[i].create(NSTATES);
		bel.belEdges[i].create(NSTATES, NSTATES);
	}
	dMatrix Mi_HH(NSTATES, NSTATES);
	dVector Ri_H (NSTATES);
	dVector Ri_Y;
	dVector Pi_Y;
	dMatrix Mi_YH;
	dVector alpha_H(NSTATES);
	dVector newAlpha_H(NSTATES);
	dVector tmp_H(NSTATES);
	bel.belStates[NNODES-1].set(0.0);
	for (int i = NNODES-1; i > 0; i--) {
		// compute the Mi matrix
		computeLogMi(fGen, model, X, i, -1, Mi_HH, Ri_H, false, 
					 false);
		computeObsMsg(fGen, model, X, i, Mi_YH, Ri_Y, Pi_Y, false, 
					  bUseStatePerNodes);
		tmp_H.set(bel.belStates[i]);
		tmp_H.add(Ri_Y);
		tmp_H.add(Ri_H);
		LogMultiply(Mi_HH, tmp_H, bel.belStates[i-1]);
	}
	
	for (int i = 0; i < NNODES; i++) {
		// compute the Mi matrix
		computeLogMi(fGen, model, X, i, -1, Mi_HH, Ri_H, false, 
					 false);
		computeObsMsg(fGen, model, X, i, Mi_YH, Ri_Y, Pi_Y, false, 
					  bUseStatePerNodes);
		// newAlpha_H = Ri_H + messages from H_i-1
		if (i > 0) {
			tmp_H.set(alpha_H);
			Mi_HH.transpose();
			LogMultiply(Mi_HH, tmp_H, newAlpha_H);
			newAlpha_H.add(Ri_H);
		}
		else {
			newAlpha_H.set(Ri_H);
		}
		// We have now to compute the Y-H beliefs
		Mi_YH.transpose();
		tmp_H.set(bel.belStates[i]);
		tmp_H.add(newAlpha_H);
		LogMultiply(Mi_YH, tmp_H, bel.belStates[i+NNODES]);
		for(int yprev = 0; yprev < NLABELS; yprev++)
				for(int yp = 0; yp < NSTATES; yp++)
					bel.belEdges[i+NNODES](yprev,yp) += tmp_H[yp] + Pi_Y[yprev];
		bel.belEdges[i+NNODES].add(Mi_YH);
		newAlpha_H.add(Ri_Y);
		if (i > 0) {
			tmp_H.set(Ri_H);
			tmp_H.add(bel.belStates[i]);
			tmp_H.add(Ri_Y);
			tmp_H.transpose();
			for(int yprev = 0; yprev < NSTATES; yprev++)
				for(int yp = 0; yp < NSTATES; yp++)
					bel.belEdges[i-1](yprev,yp) += tmp_H[yp] + alpha_H[yprev];

			Mi_HH.transpose();
			bel.belEdges[i-1].add(Mi_HH);
		}
		// We update bel.belStates[i] to the sum of all the messages.
		bel.belStates[i].add(newAlpha_H);
		alpha_H.set(newAlpha_H);
	}
	double lZx = alpha_H.logSumExp();
	for (int i = 0; i < 2*NNODES; i++)
	{
		bel.belStates[i].add(-lZx);
		bel.belStates[i].eltExp();
		bel.belEdges[i].add(-lZx);
		bel.belEdges[i].eltExp();
	}
	bel.partition = lZx;
}

void InferenceEngineDC::computeObsMsg(FeatureGenerator* fGen, Model* model,
									  DataSequence* X, int i, dMatrix& Mi_YH, 
									  dVector& Ri_Y, dVector& Pi_Y,  bool takeExp, 
									  bool bUseStatePerNodes)
{
#if !defined(_VEC_FEATURES) && !defined(_OPENMP)
	featureVector* vecFeatures;
#endif
#if defined(_OPENMP)
	int ThreadID = omp_get_thread_num();
	if (ThreadID >= nbThreadsMP)
		ThreadID = 0;
#else
	int ThreadID = 0;
#endif

	// This function compute Mi_YH = Sum simply the matrices between the hidden
	// states and the messages. It also return Ri_Y, which is the messages going
	// from Y to H if Y is not know.
	int NSTATES = model->getNumberOfStates();
	int NLABELS = model->getNumberOfStateLabels();
	Mi_YH.create(NLABELS, NSTATES);
	Ri_Y.create(NSTATES);
	dVector* lambda=model->getWeights(-1);
	// We also need to get the messages from the labels
#if defined(_VEC_FEATURES) || defined(_OPENMP)
	fGen->getFeatures(vecFeaturesMP[ThreadID], X, model, 0, X->length(), 0);
	feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
	for (int j = 0; j < vecFeaturesMP[ThreadID].size() ; j++, pFeature++) {
#else
	vecFeatures = fGen->getFeatures(X, model, 0, X->length(), 0);
	feature* pFeature = vecFeatures->getPtr();
	for (int j = 0; j < vecFeatures->size() ; j++, pFeature++) {
#endif
		int f = pFeature->id;
		int yp = pFeature->nodeState;
		int yprev = pFeature->prevNodeState;
		double val = pFeature->value;
		Mi_YH(yp, yprev) += (*lambda)[f]*val;
    }
	if (takeExp) {
		if (bUseStatePerNodes){
			Pi_Y.create(NLABELS, COLVECTOR);
			Pi_Y[X->getStateLabels(i)] = 1.0;
		} else {
			Pi_Y.create(NLABELS, COLVECTOR, 1.0);
		}
		Mi_YH.eltExp();
		Ri_Y.multiply(Mi_YH, Pi_Y);
	} else {
		if (bUseStatePerNodes){
			for(int j=0; j<NSTATES;j++){
				Ri_Y[j] = Mi_YH(j, X->getStateLabels(i));
			}
			Pi_Y.create(NLABELS, COLVECTOR, -INF_VALUE);
			Pi_Y[X->getStateLabels(i)] = 0.0;
			
		} else{
			Pi_Y.create(NLABELS, COLVECTOR, 0.0);
			LogMultiply(Mi_YH, Pi_Y, Ri_Y);
		}
	}
}
