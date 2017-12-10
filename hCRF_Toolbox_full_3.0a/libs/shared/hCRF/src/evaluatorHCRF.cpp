//-------------------------------------------------------------
// Hidden Conditional Random Field Library - EvaluatorHCRF
// Component
//
//	May 1st, 2006

#include "evaluator.h"
#include <assert.h>
using namespace std;

/////////////////////////////////////////////////////////////////////
// Evaluator HCRF Class
/////////////////////////////////////////////////////////////////////

// *
// Constructor and Destructor
// *

EvaluatorHCRF::EvaluatorHCRF() : Evaluator()
{
}

EvaluatorHCRF::EvaluatorHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen) : Evaluator(infEngine, featureGen)
{
}

EvaluatorHCRF::~EvaluatorHCRF()
{

}

// *
// Public Methods
// *

//computes OVERALL error of the datasequence
double EvaluatorHCRF::computeError(DataSequence* X, Model* m, bool bComputeMaxMargin)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In EvaluatorHCRF::computeError");
	}

	int nbSeqLabels = m->getNumberOfSequenceLabels(); 
	dVector Partition(nbSeqLabels);

	for(int y=0; y<nbSeqLabels; y++) 
		Partition[y] = pInfEngine->computePartition(
				pFeatureGen,X,m,y,false,bComputeMaxMargin);

	if( !bComputeMaxMargin ) {
		//return log(Sum_y' Z(y'|x)) - log(Z(y|x)) 
		return Partition.logSumExp() - Partition[X->getSequenceLabel()];
	}
	else {
		//return max_y' log(Z(y'|x)) - log(Z(y|x))
		int max_y=-1; double max_val=-DBL_MAX;
		for(int y=0; y<nbSeqLabels; y++) {
			if( Partition[y] > max_val ) {
				max_y=y; max_val = Partition[y];
			}
		}
		return max_val - Partition[X->getSequenceLabel()];
	}
}


int EvaluatorHCRF::computeSequenceLabel(DataSequence* X, Model* m, dMatrix * probabilities, bool bComputeMaxMargin)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In EvaluatorHCRF::computeSequenceLabel");
	}

	Beliefs bel;
	int labelCounter=0;
	int numberofSequenceLabels = m->getNumberOfSequenceLabels();
	double partition = 0;
	double bestScore = -100000000;
	int bestLabel = -1;
	if(probabilities)
		probabilities->create(1,numberofSequenceLabels);

	//Compute the State Labels i.e.
	for(labelCounter=0;labelCounter<numberofSequenceLabels;labelCounter++){
		partition = pInfEngine->computePartition(pFeatureGen, X, m,labelCounter,false,bComputeMaxMargin);
		probabilities->setValue(labelCounter,0, partition);
		if(bestScore<partition){
			bestScore = partition;
			bestLabel = labelCounter;
		}
	}
	return bestLabel;
}

