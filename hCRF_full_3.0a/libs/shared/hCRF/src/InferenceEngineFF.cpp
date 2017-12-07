#include "inferenceengine.h"
#include <assert.h>

InferenceEngineFF::InferenceEngineFF(int inDelay)
	: InferenceEngine()
{
	delay = inDelay;
}


int InferenceEngineFF::computeSingleBelief(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob)
{	
	int NSTATES = model->getNumberOfStates();
	int NNODES = dataSequence->length();
	int pos = dataSequence->getPosition();
	int windowSize = dataSequence->getWindowSize();
	
	//Compute Mi_YY, in our case Mi_YY are the same for all positions
	dMatrix Mi_YY(NSTATES,NSTATES);
	dVector Ri_Y(NSTATES);
	computeLogMi(fGen, model, dataSequence, 1, -1, Mi_YY, Ri_Y, false, false);
	
	dMatrix Mi_YY2(NSTATES,NSTATES);
	dVector tmp_Y(NSTATES);

	// Update Alpha
	dVector* alpha;
	if(dataSequence->getAlpha() == 0)
	{				
		dataSequence->initializeAlpha(NSTATES);
		alpha = dataSequence->getAlpha();
		computeLogMi(fGen, model, dataSequence, (pos+windowSize) % NNODES, -1, Mi_YY2, Ri_Y, false, false);
		alpha->set(Ri_Y);
	}
	else
	{
		computeLogMi(fGen, model, dataSequence, (pos+windowSize) % NNODES, -1, Mi_YY2, Ri_Y, false, false);
		alpha = dataSequence->getAlpha();
		tmp_Y.set(*alpha);
		Mi_YY.transpose();
		LogMultiply(Mi_YY, tmp_Y, *alpha);
		alpha->add(Ri_Y);
	}

	// Calculate beta for node in pos
	dVector beta(NSTATES);
	beta.set(0);
	Mi_YY.transpose();
	for(int i=pos+NNODES-1; i>pos+windowSize; i--)
	{
		int index = i%NNODES;
		computeLogMi(fGen, model, dataSequence, index, -1, Mi_YY2, Ri_Y, false, false);
		tmp_Y.set(beta);
		tmp_Y.add(Ri_Y);
		LogMultiply(Mi_YY,tmp_Y,beta);
	}		
	
	// Calculate probability distribution

	beta.add(*alpha);	
	double LZx = beta.logSumExp();
	beta.add(-LZx);
	beta.eltExp();
	
	int nbLabel = 2;
	int nbHiddenNodes = NSTATES/nbLabel;
	prob->set(0);
	for(int i =0; i<nbLabel; i++)
		for(int j=0; j<nbHiddenNodes; j++){
			prob->addValue(i, beta.getValue(i*nbHiddenNodes +j));
	}

	//Not really used anywhere...
	return 1;
}




double InferenceEngineFF::computePartition(FeatureGenerator* fGen,
											 DataSequence* X, Model* model,
											 int seqLabel,
											 bool bUseStatePerNodes,
											 bool bMaxProduct)
{
	if( bMaxProduct )
		throw HcrfNotImplemented("InferenceEngineFF for max-margin is not implemented");

	Beliefs bel;
	computeBeliefsLog(bel, fGen,X, model, true,seqLabel, bUseStatePerNodes);
	return bel.partition;
}

void InferenceEngineFF::computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,
									  DataSequence* X, Model* model,
									  int bComputePartition,
									  int seqLabel, bool bUseStatePerNodes, bool bMaxProduct)
{
	if( bMaxProduct )
		throw HcrfNotImplemented("InferenceEngineFF for max-margin is not implemented");

	//Simulate a real-time sequence, forward all alphas in one batch, but propagate betas only over
	// a window of length delay.
	computeBeliefsLog(bel, fGen,X, model, bComputePartition, seqLabel, bUseStatePerNodes);
}

void InferenceEngineFF::computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen,
										  DataSequence* X, Model* model,
										  int, int seqLabel,
										  bool bUseStatePerNodes)
{
	if(model->getAdjacencyMatType()!=CHAIN){
		throw HcrfBadModel("InferenceEngineFF need a model based on a Chain");
	}
	int NNODES=X->length();
	int NSTATES = model->getNumberOfStates();
	bel.belStates.resize(NNODES);

	for(int i=0;i<NNODES;i++)
	{
		bel.belStates[i].create(NSTATES);
	}
	int NEDGES = NNODES-1;
	bel.belEdges.resize(NEDGES);
	for(int i=0;i<NEDGES;i++)
	{
		bel.belEdges[i].create(NSTATES,NSTATES, 0);
	}
	dMatrix Mi_YY (NSTATES,NSTATES);
	dVector Ri_Y (NSTATES);
	dVector alpha_Y(NSTATES);
	dVector beta (NSTATES);
	dVector newAlpha_Y(NSTATES);
	dVector tmp_Y(NSTATES);
	dVector partitionValues(NNODES);

	// compute beta values in pseudo-backward scans.
	// also scale beta-values to 1 to avoid numerical problems.
	bel.belStates[NNODES-1].set(0);
	//JCL: First make this work, then see if there are optimizations to make...
	//The last betas should be computed separately in the 'regular' way.
	for (int i = NNODES-1; i > NNODES-delay; i--)
	{
		// compute the Mi matrix
		computeLogMi(fGen, model, X, i, seqLabel, Mi_YY, Ri_Y, false,
					 bUseStatePerNodes);
		tmp_Y.set(bel.belStates[i]);
		tmp_Y.add(Ri_Y);
		LogMultiply(Mi_YY,tmp_Y,bel.belStates[i-1]);
	}
	//Compute other betas in a short backward sweep of length delay.
	for (int i = 0; i < NNODES-delay; i++)
	{
		beta.set(0);
		for(int j=i+delay-1; j > i; j--)
		{
			// compute the Mi matrix
			computeLogMi(fGen, model, X, j, seqLabel, Mi_YY, Ri_Y, false,
					 bUseStatePerNodes);
			tmp_Y.set(beta);
			tmp_Y.add(Ri_Y);
			LogMultiply(Mi_YY,tmp_Y,beta);
		}
		bel.belStates[i].set(beta);
	}

	// Compute Alpha values
	alpha_Y.set(0);
	for (int i = 0; i < NNODES; i++) {
		// compute the Mi matrix
		computeLogMi(fGen,model, X, i, seqLabel, Mi_YY, Ri_Y,false,
					 bUseStatePerNodes);
		if (i > 0)
		{
			tmp_Y.set(alpha_Y);
			Mi_YY.transpose();
			LogMultiply(Mi_YY, tmp_Y, newAlpha_Y);
			newAlpha_Y.add(Ri_Y);
		}
		else
		{
			newAlpha_Y.set(Ri_Y);
		}
		if (i > 0)
		{
			tmp_Y.set(Ri_Y);
			tmp_Y.add(bel.belStates[i]);
			Mi_YY.transpose();
			bel.belEdges[i-1].set(Mi_YY);
			for(int yprev = 0; yprev < NSTATES; yprev++)
				for(int yp = 0; yp < NSTATES; yp++)
					bel.belEdges[i-1](yprev,yp) += tmp_Y[yp] + alpha_Y[yprev];
		}
	  
		bel.belStates[i].add(newAlpha_Y);
		alpha_Y.set(newAlpha_Y);
		partitionValues[i] = bel.belStates[i].logSumExp();
	}

	double lZx = alpha_Y.logSumExp();
	for (int i = 0; i < NNODES; i++)
	{
		//bel.belStates[i].add(-lZx);
		bel.belStates[i].add(-partitionValues[i]);
		bel.belStates[i].eltExp();
	}
	double lEdgePart = 0;
	for (int i = 0; i < NEDGES; i++)
	{
		//This is slower, but at least it works.
		lEdgePart = 0;
		for (int j = 0; j < NSTATES; j++)
		{
			bel.belEdges[i].getCol(j,tmp_Y);
			lEdgePart += tmp_Y.logSumExp();
		}
		bel.belEdges[i].add(-lEdgePart);

		//This doesn't work. This partition value is different, somehow?
		//bel.belEdges[i].add(-partitionValues[i]);

		bel.belEdges[i].eltExp();
	}

	//Return the global partition value.
	bel.partition = lZx;
}
