//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of 
// EvaluatorMVLDCRF Component
//
// Yale Song (yalesong@csail.mit.edu)
// October, 2011

#include "evaluator.h"
#include <assert.h> 

/////////////////////////////////////////////////////////////////////
// EvaluatorMVLDCRF Class
/////////////////////////////////////////////////////////////////////

EvaluatorMVLDCRF::EvaluatorMVLDCRF(): Evaluator()
{}

EvaluatorMVLDCRF::EvaluatorMVLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) 
: Evaluator(infEngine, featureGen)
{}

EvaluatorMVLDCRF::~EvaluatorMVLDCRF()
{}


// Computes OVERALL error given X
double EvaluatorMVLDCRF::computeError(DataSequence* X, Model* m, bool bComputeMaxMargin)
{
	if(!pInfEngine || !pFeatureGen)
		throw HcrfBadPointer("In EvaluatorMVLDCRF::computeError");
 
	double partition = pInfEngine->computePartition(pFeatureGen, X, m, -1, false, bComputeMaxMargin);
	double partitionMasked = pInfEngine->computePartition(pFeatureGen, X, m, -1, true, bComputeMaxMargin);

	// return log(Z(h|x)) - log(Z(h*|x)) 
	return partition - partitionMasked;
}


// Compute the probability of each nodes in the datasequence given the model. 
// Returns a label vector and a probability matrix
void EvaluatorMVLDCRF::computeStateLabels(
	DataSequence* X, Model* m, iVector* vecStateLabels, dMatrix* prob, bool bComputeMaxMargin)
{
	if(!pInfEngine || !pFeatureGen)
		throw HcrfBadPointer("EvaluatorMVLDCRF::computeStateLabels");

	Beliefs bel;
	pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, false, -1, false, bComputeMaxMargin);

	int nbNodes = (int) bel.belStates.size();
	int nbLabels = m->getNumberOfStateLabels();
	int seqLength = X->length();
	int nbViews = m->getNumberOfViews();

	vecStateLabels->create(seqLength);
	if( prob ) prob->create(seqLength, nbLabels);

	dMatrix sumBeliefsPerLabel(nbLabels,nbViews);
	dVector avgBeliefsPerLabel(nbLabels);

	// Belief at each frame is computed as an average of multi-view beliefs
	for(int xt=0; xt<seqLength; xt++) {
		sumBeliefsPerLabel.set(0);
		avgBeliefsPerLabel.set(0);
		// Sum up beliefs
		for(int v=0; v<nbViews; v++) {
			int xi = v*seqLength + xt;
			for(int h=0; h<m->getNumberOfStatesMV(v); h++)
				sumBeliefsPerLabel(v,m->getLabelPerStateMV(v)[h]) += bel.belStates[xi][h];
		}
		// Compute the average beliefs across views
		for(int y=0; y<nbLabels; y++) 
			avgBeliefsPerLabel[y] = sumBeliefsPerLabel.colSum(y) / nbViews;
		// Find the max belief
		int yt = 0;
		vecStateLabels->setValue(xt,yt);
		double maxBelief = avgBeliefsPerLabel[yt];
		if( prob ) prob->setValue(yt,xt,avgBeliefsPerLabel[yt]);
		for( yt=1; yt<nbLabels; yt++ ) {
			if( prob ) prob->setValue(yt,xt,avgBeliefsPerLabel[yt]);
			if( maxBelief < avgBeliefsPerLabel[yt] ) {
				vecStateLabels->setValue(xt,yt);
				maxBelief = avgBeliefsPerLabel[yt];
			}
		}
	}
}
