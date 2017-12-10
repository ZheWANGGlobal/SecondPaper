//-------------------------------------------------------------
// Hidden Conditional Random Field Library - EvaluatorLVPERCEPTRON
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

EvaluatorLVPERCEPTRON::EvaluatorLVPERCEPTRON() 
	: Evaluator()
{

}

EvaluatorLVPERCEPTRON::~EvaluatorLVPERCEPTRON() 
{

}


EvaluatorLVPERCEPTRON::EvaluatorLVPERCEPTRON(InferenceEngine* infEngine, FeatureGenerator* featureGen) 
	: Evaluator(infEngine, featureGen)
{

}


// *
// Public Methods
// *

//computes OVERALL error of the datasequence
double EvaluatorLVPERCEPTRON::computeError(DataSequence* X, Model* m, bool bComputeMaxMargin)
{
	return 0;
}


//computes OVERALL error of the dataset
double EvaluatorLVPERCEPTRON::computeError(DataSet* X, Model* m, bool bComputeMaxMargin)
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("In EvaluatorLVPERCEPTRON::computeError");
	}

	double error = 0;
	double ZZ = 0;
	int NumIters = (int)X->size();
	int i = 0;

	for(i = 0; i<NumIters; i++)
	{
		error += computeError(X->at(i),m);
	}

	// what is this????
	if(m->getRegL2Sigma() != 0.0f)
	{
	   double weightNorm = m->getWeights()->l2Norm(false);
	   error += weightNorm / (2.0*m->getRegL2Sigma()*m->getRegL2Sigma());

	}
	return error;
}

void EvaluatorLVPERCEPTRON::computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, dMatrix * probabilities, bool bComputeMaxMargin)
/*
This function is used to compute the probability of each nodes in the
datasequence given the model. It return a vector of labels and the matrix
of probability
*/
{
	if(!pInfEngine || !pFeatureGen){
		throw HcrfBadPointer("EvaluatorLVPERCEPTRON::computeStateLabels");
	}
	//Beliefs bel;
	//pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, false);

	//int nbNodes = (int)bel.belStates.size();
	//int nbLabels = m->getNumberOfStateLabels();
	//int nbStates = 0;
	//if(nbNodes > 0)
	//	nbStates = bel.belStates[0].getLength();

	//vecStateLabels->create(nbNodes);
	//if(probabilities)
	//	probabilities->create(nbNodes,nbLabels);

	//dVector sumBeliefsPerLabel(nbLabels);

	//for(int n = 0; n<nbNodes; n++) 
	//{
	//	sumBeliefsPerLabel.set(0);
	//	// Sum beliefs
	//	//TODO: Take into account the shared state (maybe through a weighted sumation)
	//	for (int s = 0; s<nbStates; s++) 
	//	{
	//		sumBeliefsPerLabel[m->getLabelPerState()[s]] += bel.belStates[n][s];
	//	}
	//	// find max value
	//	vecStateLabels->setValue(n, 0);
	//	double MaxBel = sumBeliefsPerLabel[0];
	//	if(probabilities)
	//		probabilities->setValue(0,n,sumBeliefsPerLabel[0]);
	//	for (int cur_label = 1; cur_label < nbLabels; cur_label++) 
	//	{
	//		if(probabilities)
	//			probabilities->setValue(cur_label, n, sumBeliefsPerLabel[cur_label]);
	//		if(MaxBel < sumBeliefsPerLabel[cur_label]) {
	//			vecStateLabels->setValue(n, cur_label);
	//			MaxBel = sumBeliefsPerLabel[cur_label];
	//		}
	//	}
	//}
}
