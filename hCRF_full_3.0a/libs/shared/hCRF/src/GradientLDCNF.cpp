//-------------------------------------------------------------
// Hidden Conditional Random Field Library
// GradientLDCNF Component
//
// Julien-Charles Lévesque
// July 28th, 2011

#include "gradient.h"
#include "GateNodeFeatures.h"

using namespace std;

GradientLDCNF::GradientLDCNF(InferenceEngine* infEngine,
							 FeatureGenerator* featureGen):
Gradient(infEngine, featureGen)
{
}

double GradientLDCNF::computeGradient(dVector& vecGradient, Model* m,
									DataSequence* X, bool bComputeMaxMargin)
{
	if( bComputeMaxMargin )
		throw HcrfNotImplemented("GradientLNCRF for max-margin is not implemented");

	//These variables are all related to gates.
	dMatrix* gates = X->getGateMatrix();
	dVector* weights = m->getWeights();
	GateNodeFeatures* gateNF = (GateNodeFeatures*)pFeatureGen->getFeatureById(GATE_NODE_FEATURE_ID);
	int gateIndexOffset = gateNF->getIdOffset();
	int nbGates;
	//This number is a pain to compute, so ask it from the features.
	int nbFeaturesPerFrame = m->getNumberOfRawFeaturesPerFrame();
	int nbFeaturesPerGate = gateNF->getNbFeaturesPerGate();
	int nbStates = m->getNumberOfStateLabels();
	double gateOut;
	double weightedGateOut;
	int gate_i;
	
	//compute beliefs
	Beliefs bel;
	Beliefs belMasked;
	pInfEngine->computeBeliefs(bel,pFeatureGen, X, m, true,-1, false);
	pInfEngine->computeBeliefs(belMasked,pFeatureGen, X, m, true,-1, true);
	
	//Check the size of vecGradient
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);
		
	//Gates are now computed.
	nbGates = gates->getHeight();
	dVector probWeightSum(nbGates);

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
	//Loop over nodes to compute features and update the gradient
	for(int i = 0; i < X->length(); i++)
	{
		// Read the label for this state
		int s = X->getStateLabels(i);
		probWeightSum.set(0);
		
		//Will be used to check if current feature is a start feature.
		FeatureType* startFeatures = pFeatureGen->getFeatureById(START_FEATURE_ID);
		int startFeaturesIndex = -1;
		if (startFeatures)
			startFeaturesIndex = startFeatures->getIdOffset();
			
		//Get nodes features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,i,-1);
		// Loop over features
		feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
		for(int j = 0; j < vecFeaturesMP[ThreadID].size(); j++, pFeature++)
#else
		vecFeatures = pFeatureGen->getFeatures(X,m,i,-1);
		// Loop over features
		feature* pFeature = vecFeatures->getPtr();
		for(int j = 0; j < vecFeatures->size(); j++, pFeature++)
#endif
		{
			//p(y_i=s|x)*f_k(i,s,x)
			vecGradient[pFeature->id] -= bel.belStates[i][pFeature->nodeState]*pFeature->value;
			vecGradient[pFeature->id] += belMasked.belStates[i][pFeature->nodeState]*pFeature->value;
			
			if(startFeaturesIndex==-1 || (pFeature->id - pFeature->nodeState - startFeaturesIndex) != 0)
			{
				//Compute expectation for gates.
				gateOut = pFeature->value;
				weightedGateOut = weights->getValue(pFeature->id)*(1.-gateOut)*gateOut;
				//Simplest way to get back the gate index.
				gate_i = pFeature->id - (pFeature->nodeState*nbGates + gateIndexOffset);
				
				assert(gate_i >= 0 && gate_i < nbGates);
				
				probWeightSum[gate_i] -= bel.belStates[i][pFeature->nodeState] * weightedGateOut;
				probWeightSum[gate_i] += belMasked.belStates[i][pFeature->nodeState] * weightedGateOut;
			}
		}
	
		//Compute gate gradients.
		featureVector vecPreGateFeatures;
		gateNF->getPreGateFeatures(vecPreGateFeatures, X, m, i, -1);
		pFeature = vecPreGateFeatures.getPtr();

		for(int j = 0; j < vecPreGateFeatures.size(); j++, pFeature++)
		{
			//Gate index is stored in nodeState, it is not used anyways.
			gate_i = pFeature->nodeState;
			//Ll minus expectation term.
			vecGradient[pFeature->id] += pFeature->value*probWeightSum[gate_i];
		}
	}

	//Loop over edges
	for(int i = 0; i < X->length()-1; i++) // Loop over all rows (the previous node index)
	{
		//Get nodes features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,i+1,i);
		// Loop over features
		feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
		for(int j = 0; j < vecFeaturesMP[ThreadID].size(); j++, pFeature++)
#else
		vecFeatures = pFeatureGen->getFeatures(X,m,i+1,i);
		// Loop over features
		feature* pFeature = vecFeatures->getPtr();
		for(int j = 0; j < vecFeatures->size(); j++, pFeature++)
#endif
		{
			//p(y_i=s1,y_j=s2|x)*f_k(i,j,s1,s2,x) is subtracted from the gradient 
			vecGradient[pFeature->id] -= bel.belEdges[i](pFeature->prevNodeState,pFeature->nodeState)*pFeature->value;
			vecGradient[pFeature->id] += belMasked.belEdges[i](pFeature->prevNodeState,pFeature->nodeState)*pFeature->value;
		}
	}

	//Return -log instead of log() [Moved to Gradient::ComputeGradient by LP]
	//vecGradient.negate();
	return bel.partition - belMasked.partition;
}

double GradientLDCNF::computeGradient(dVector& vecGradient, Model* m, DataSet* X)
{
	double ans = Gradient::computeGradient(vecGradient,m,X);
	
	//Add R_hg regularization part.
	double alphaRegL1 = m->getAlphaRegL1();
	double alphaRegL2 = m->getAlphaRegL2();
	
	//Find beginning of hg weights
	GateNodeFeatures* gateF = (GateNodeFeatures*)pFeatureGen->getFeatureById(GATE_NODE_FEATURE_ID);
	
	int nbGates = gateF->getNbGates();
	int nbHiddenStates = m->getNumberOfStates();
	dVector* w = m->getWeights();
	
	if (gateF && (alphaRegL1 != 0))
	{
		double temp = 0.;
		
		int gateFeatureStart = gateF->getIdOffset();
		for (int i = 0; i < nbHiddenStates; i++)
		{
			for (int j = 0; j < nbGates; j++)
			{
				temp = 0.;
				for(int k = 0; k < nbHiddenStates; k++)
				{
					if (k!=i)
						temp += (*w)[gateFeatureStart + k*nbGates + j];
				}
				vecGradient[gateFeatureStart + i*nbGates + j] += alphaRegL1*temp;
			}
		}
	}
	else if(gateF && (alphaRegL2 != 0))
	{
		double temp = 0.;
		
		int gateFeatureStart = gateF->getIdOffset();
		for (int i = 0; i < nbHiddenStates; i++)
		{
			for (int j = 0; j < nbGates; j++)
			{
				temp = 0.;
				for(int k = 0; k < nbHiddenStates; k++)
				{
					if (k!=i)
					{
						for(int g = 0; g < nbGates; g++)
							temp +=  (*w)[gateFeatureStart + i*nbGates + g] * (*w)[gateFeatureStart + k*nbGates + g];
						temp *= 2*(*w)[gateFeatureStart + k*nbGates + j];
					}
				}
				vecGradient[gateFeatureStart + i*nbGates + j] += alphaRegL2*temp;
			}
		}
	}
	
	return ans;
}

