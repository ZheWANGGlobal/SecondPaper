//-------------------------------------------------------------
// Hidden Conditional Random Field Library - EvaluatorMVHCRF
// Component
//
//	May 1st, 2006

#include "evaluator.h"
#include <assert.h>
using namespace std;

/////////////////////////////////////////////////////////////////////
// Evaluator MVHCRF Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

EvaluatorMVHCRF::EvaluatorMVHCRF() 
: Evaluator()
{}

EvaluatorMVHCRF::EvaluatorMVHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) 
: Evaluator(infEngine, featureGen)
{}

EvaluatorMVHCRF::~EvaluatorMVHCRF()
{}

// *
// Public Methods
// * 
double EvaluatorMVHCRF::computeError(DataSequence* X, Model* m, bool bComputeMaxMargin)
{  
	if(!pInfEngine || !pFeatureGen) 
		throw HcrfBadPointer("In EvaluatorHCRF::computeError()");

	double groundTruthLabel = X->getSequenceLabel();
	double groundTruthPartition = -DBL_MAX;
	double maxPartition = -DBL_MAX;
	
	int nbSeqLabels = m->getNumberOfSequenceLabels();
	dVector Partition(nbSeqLabels);

	// For each class label, compute the partition of the data sequence, and add up all these partitions 
	for(int seqLabel=0; seqLabel<nbSeqLabels; seqLabel++) 
	{		 
		Partition[seqLabel] = pInfEngine->computePartition(pFeatureGen,X,m,seqLabel,false,bComputeMaxMargin);
		if( seqLabel==groundTruthLabel )
			groundTruthPartition = Partition[seqLabel]; 
		if( Partition[seqLabel] > maxPartition )
			maxPartition = Partition[seqLabel];
	} 

	if( bComputeMaxMargin )	
		return maxPartition - groundTruthPartition; 
	else 
		return Partition.logSumExp() - groundTruthPartition; 
}


int EvaluatorMVHCRF::computeSequenceLabel(DataSequence* X, Model* m, dMatrix * probabilities, bool bComputeMaxMargin)
{ 
	if(!pInfEngine || !pFeatureGen) 
		throw HcrfBadPointer("In EvaluatorHCRF::computeSequenceLabel()");

	int nbSeqLabels = m->getNumberOfSequenceLabels();
	if( probabilities ) probabilities->create(1,nbSeqLabels);

	int bestSeqLabel = -1;
	double bestSeqScore = -DBL_MAX;
	double partition = 0;
 
	for(int seqLabel=0; seqLabel<nbSeqLabels; seqLabel++) {
		partition = pInfEngine->computePartition(pFeatureGen,X,m,seqLabel,false,bComputeMaxMargin); 		
		probabilities->setValue(seqLabel,0,partition);
		if( bestSeqScore < partition ) {
			bestSeqScore = partition;
			bestSeqLabel = seqLabel;
		}
	} 
	return bestSeqLabel;
}

