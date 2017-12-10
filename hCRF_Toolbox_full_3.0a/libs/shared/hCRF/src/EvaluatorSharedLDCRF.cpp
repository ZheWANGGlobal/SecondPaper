#include <assert.h>
#include "evaluator.h"

#define INF_VALUE 1e100

		
void EvaluatorSharedLDCRF::computeStateLabels(DataSequence* X,
											  Model* m,
											  iVector* vecStateLabels,
											  dMatrix * probabilities,
											  bool bComputeMaxMargin)
/*
This function is used to compute the probability of each nodes in the
datasequence given the model. It return a vector of labels and the matrix of
probability
*/
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("EvaluatorLDCRF::computeStateLabels");
	}
	Beliefs bel;
	pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, false, 0);

	int nbNodes = X->length();
	int nbLabels = m->getNumberOfStateLabels();
	int nbStates = 0;
	if(nbNodes > 0)
		nbStates = bel.belStates[0].getLength();

	vecStateLabels->create(nbNodes);
	if(probabilities)
		probabilities->create(nbNodes,nbLabels);

	for(int n = 0; n<nbNodes; n++)
	{
 		// find max value
		vecStateLabels->setValue(n, 0);
		double MaxBel = bel.belStates[n+nbNodes][0];
		if(probabilities)
			probabilities->setValue(0,n,bel.belStates[n+nbNodes][0]);		
		for (int cur_label = 1; cur_label < nbLabels; cur_label++)
		{
			if(probabilities)
				probabilities->setValue(cur_label, n, bel.belStates[n+nbNodes][cur_label]);
			if(MaxBel < bel.belStates[n+nbNodes][cur_label]) {
				vecStateLabels->setValue(n, cur_label);
				MaxBel = bel.belStates[n+nbNodes][cur_label];
			}
		}
	}
}


