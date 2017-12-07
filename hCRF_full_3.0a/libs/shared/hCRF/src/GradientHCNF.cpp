//-------------------------------------------------------------
// Hidden Conditional Random Field Library
// GradientHCNF Component
//
// Julien-Charles Lévesque
// July 28th, 2011

#include "gradient.h"
#include "GateNodeFeatures.h"

using namespace std;

GradientHCNF::GradientHCNF(InferenceEngine* infEngine,
							 FeatureGenerator* featureGen):
Gradient(infEngine, featureGen)
{
}

double GradientHCNF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin)
{
	//These variables are all related to gates.
	dMatrix* gates = X->getGateMatrix();
	dVector* weights = m->getWeights();
	GateNodeFeatures* gateNF = (GateNodeFeatures*)pFeatureGen->getFeatureById(GATE_NODE_FEATURE_ID);
	int gateFeatureOffset = gateNF->getIdOffset();
	int labelEdgeFeatureOffset = pFeatureGen->getFeatureById(LABEL_EDGE_FEATURE_ID)->getIdOffset();
	int nbGates;
	//This number is a pain to compute, so ask it from the features.
	int nbFeaturesPerFrame = m->getNumberOfRawFeaturesPerFrame();
	int nbFeaturesPerGate = gateNF->getNbFeaturesPerGate();
	int nbStates = m->getNumberOfStateLabels();
	double gateOut;
	double weightedGateOut;
	int gate_i;
	
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	int NumSeqLabels=m->getNumberOfSequenceLabels();
	//Get adjency matrix
	uMatrix adjMat;
	m->getAdjacencyMatrix(adjMat, X);
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);
	dVector Partition;
	Partition.resize(1,NumSeqLabels);
	std::vector<Beliefs> ConditionalBeliefs(NumSeqLabels);

	// Step 1 : Run Inference in each network to compute marginals conditioned on Y
	for(int i=0;i<NumSeqLabels;i++)
	{
		pInfEngine->computeBeliefs(ConditionalBeliefs[i],pFeatureGen, X, m, true,i);
		Partition[i] = ConditionalBeliefs[i].partition;
	}
	double f_value = Partition.logSumExp() - Partition[X->getSequenceLabel()];
	
	//gates.
	nbGates = gates->getHeight();
	
	// Step 2: Compute expected values for feature nodes conditioned on Y
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
	double value;
	double pwvalue;
	dMatrix CEValues;
	CEValues.resize(nbFeatures, NumSeqLabels);
	dVector probWeightSum(nbGates);
	//Loop over nodes to compute features and update the gradient
	for(int j=0;j<NumSeqLabels;j++) { //For every label
		//Reset vector between every iteration.
		probWeightSum.set(0);
		for(int i = 0; i < X->length(); i++) { //For every node
#if defined(_VEC_FEATURES) || defined(_OPENMP)
			pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X, m, i, -1, j);
			// Loop over features
			feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
			for(int k = 0; k < vecFeaturesMP[ThreadID].size(); k++, pFeature++)
#else
			vecFeatures = pFeatureGen->getFeatures(X, m, i, -1, j);
			// Loop over features
			feature* pFeature = vecFeatures->getPtr();
			for(int k = 0; k < vecFeatures->size(); k++, pFeature++)
#endif
			{
                //p(s_i=s|x,Y) * f_k(i,s,x,y) 
				value = ConditionalBeliefs[j].belStates[i][pFeature->nodeState] * pFeature->value;
				CEValues.setValue(j, pFeature->globalId, CEValues(j, pFeature->globalId) + value); // one row for each Y
				
				if (pFeature->featureTypeId == GATE_NODE_FEATURE_ID)
				{
					//Compute expectation for gates.
					gateOut = pFeature->value;
					weightedGateOut = weights->getValue(pFeature->globalId)*(1.-gateOut)*gateOut;
					//Simplest way to get back the gate index.
					gate_i = pFeature->globalId - (pFeature->nodeState*nbGates + gateFeatureOffset);
					
					assert(gate_i >= 0 && gate_i < nbGates);
					
					pwvalue = ConditionalBeliefs[j].belStates[i][pFeature->nodeState] * weightedGateOut;
					probWeightSum[gate_i] += pwvalue;
				}
	
			} // end for every feature
			
			//Compute gate expectations.
			featureVector vecPreGateFeatures;
			gateNF->getPreGateFeatures(vecPreGateFeatures, X, m, i, -1, j);
			pFeature = vecPreGateFeatures.getPtr();

			for(int k = 0; k < vecPreGateFeatures.size(); k++, pFeature++)
			{
				//Gate index is stored in nodeState, it is not used anyways.
				gate_i = pFeature->nodeState;
				CEValues.setValue(j,pFeature->globalId, CEValues(j,pFeature->globalId) + pFeature->value*probWeightSum[gate_i]);
			} // end for every pregate feature
		} // end for every node
	
		/*
		//Compute gate weights
		for (int k = 0; k < nbGates*)
		
		for(int i = 0; i < X->length(); i++)
		{
			vecFeatures = pFeatureGen->getPreGateFeatures(X, m, i, -1, j);
			// Loop over features
			feature* pFeature = vecFeatures->getPtr();
			//for(int k = 0; k < vecFeatures->size(); k++, pFeature++)
		}*/
	} // end for ever Sequence Label
	
	
	
	// Step 3: Compute expected values for edge features conditioned on Y
	//Loop over edges to compute features and update the gradient
	for(int j=0;j<NumSeqLabels;j++){
		int edgeIndex = 0;
	    for(int row = 0; row < X->length(); row++){
			// Loop over all rows (the previous node index)
		    for(int col = row; col < X->length() ; col++){
				//Loop over all columns (the current node index)
				if(adjMat(row,col) == 1) {
					//Get nodes features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
					pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,col,row,j);
					// Loop over features
					feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
					for(int k = 0; k < vecFeaturesMP[ThreadID].size(); k++, pFeature++)
#else
					vecFeatures = pFeatureGen->getFeatures(X,m,col,row,j);
					// Loop over features
					feature* pFeature = vecFeatures->getPtr();
					for(int k = 0; k < vecFeatures->size(); k++, pFeature++)
#endif
					{
                        //p(y_i=s1,y_j=s2|x,Y)*f_k(i,j,s1,s2,x,y) 
						value = ConditionalBeliefs[j].belEdges[edgeIndex](pFeature->prevNodeState,pFeature->nodeState) * pFeature->value;
						CEValues.setValue(j,pFeature->globalId, CEValues(j,pFeature->globalId) + value);
					}
					edgeIndex++;
				}
			}
		}
	}
	
	// Step 4: Compute Joint Expected Values
	dVector JointEValues;
	JointEValues.resize(1,nbFeatures);
	JointEValues.set(0);
	dVector rowJ;
	rowJ.resize(1,nbFeatures);
	dVector GradientVector;
	double sumZLog=Partition.logSumExp();
	for (int j=0;j<NumSeqLabels;j++)
	{
        CEValues.getRow(j, rowJ);
		rowJ.multiply(exp(Partition.getValue(j)-sumZLog));
		JointEValues.add(rowJ);
	}
	
	// Step 5 Compute Gradient as Exi[i,*,*] -Exi[*,*,*], that is difference
	// between expected values conditioned on Sequence Labels and Joint expected
	// values
	CEValues.getRow(X->getSequenceLabel(), rowJ); // rowJ=Expected value
												  // conditioned on Sequence
												  // label Y

	// [Negation moved to Gradient::ComputeGradient by LP]
	//	 rowJ.negate(); 
	JointEValues.negate();
	rowJ.add(JointEValues);
	vecGradient.add(rowJ);
	return f_value;
}

double GradientHCNF::computeGradient(dVector& vecGradient, Model* m, DataSet* X)
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

