#include "GateNodeFeatures.h"

using namespace std;


GateNodeFeatures::GateNodeFeatures(int nbGates, int winSize)
: FeatureType(),
windowSize(winSize),
nbGates(nbGates)
{
	strFeatureTypeName = "Gate Node Feature Type";
	featureTypeId = GATE_NODE_FEATURE_ID;
	basicFeatureType = NODE_FEATURE;
}


void GateNodeFeatures::getFeatures(featureVector& listFeatures,
									DataSequence* X, Model* m, int nodeIndex,
									int prevNodeIndex, int seqLabel)
{
	int idNode = 0;
	int nbStateLabels = m->getNumberOfStates();
	int nbFeaturesDense = 0;
	
	dVector* lambda = m->getWeights();
	dMatrix* gates = X->getGateMatrix();

	if(X->getPrecomputedFeatures() != NULL && prevNodeIndex == -1)
	{
		dMatrix * preFeatures = X->getPrecomputedFeatures();
		nbFeaturesDense = preFeatures->getHeight();
		int nbNodes = preFeatures->getWidth();
		
		if(gates->getHeight() != nbGates || gates->getWidth() != nbNodes)
			gates->resize(nbNodes,nbGates,0.);
		
		feature* pFeature;
		
		double gateSum;
		int id;
		//First compute gate values.
		for(int g = 0; g < nbGates; g++)
		{
			idNode = 0;
			gateSum = 0;
			for(int n = nodeIndex - windowSize; n <= nodeIndex + windowSize; n++)
			{
				if(n >= 0 && n < nbNodes)
				{
					for(int f = 0; f < nbFeaturesDense; f++)
					{
						id = getIdOffset(seqLabel) + nbStateLabels*nbGates + g*nbFeaturesPerGate + idNode*nbFeaturesDense + f;
						gateSum += preFeatures->getValue(f,n) * (*lambda)[id];
					}
				}
				idNode++;
			}
			
			//Gate bias.
			gateSum += 1. * (*lambda)[getIdOffset(-1) + nbStateLabels*nbGates + g*nbFeaturesPerGate + nbFeaturesDense];
			gates->setValue(g,nodeIndex,gate(gateSum));
		}
		
		//Then compute node feature value from gate values.
		for(int s = 0; s < nbStateLabels; s++)
		{
			for(int g = 0; g < nbGates; g++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset(seqLabel) + g + s*nbGates;
				pFeature->globalId = getIdOffset() + g + s*nbGates;
				pFeature->nodeIndex = nodeIndex;
				pFeature->nodeState = s;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = -1;
				pFeature->sequenceLabel = seqLabel;
				pFeature->value = gates->getValue(g,nodeIndex);
				pFeature->featureTypeId = GATE_NODE_FEATURE_ID;
			}
		}
	}
	
	// Load Sparse raw features
	if(X->getPrecomputedFeaturesSparse() != NULL && prevNodeIndex == -1)
	{
		dMatrixSparse * preFeaturesSparse = X->getPrecomputedFeaturesSparse();
		int nbFeaturesSparse = (int)preFeaturesSparse->getHeight();
		int nbNodes = (int)preFeaturesSparse->getWidth();
		feature* pFeature;
		
		if(gates->getHeight() != nbGates || gates->getWidth() != nbNodes)
			gates->resize(nbNodes,nbGates,0.);
		
		double gateSum;
		int id;
		int f;
		int irIndex;
		size_t numElementsInCol;
		//First compute gate values.
		for(int g = 0; g < nbGates; g++)
		{
			idNode = 0;
			gateSum = 0;
			for(int n = nodeIndex - windowSize; n <= nodeIndex + windowSize; n++)
			{
				if(n >= 0 && n < nbNodes)
				{
					irIndex = (int)preFeaturesSparse->getJc()->getValue(n);
					numElementsInCol = preFeaturesSparse->getJc()->getValue(n+1) - irIndex;
					
					for(unsigned int i = 0; i < numElementsInCol; i++)
					{
						f = (int)preFeaturesSparse->getIr()->getValue(irIndex + i);// feature ID
						id = getIdOffset(seqLabel) + nbStateLabels*nbGates + g*nbFeaturesPerGate + idNode*nbFeaturesSparse + f;
						gateSum += preFeaturesSparse->getPr()->getValue(irIndex + i) * (*lambda)[id];
					}
				}
				idNode++;
			}
			
			//Gate bias.
			gateSum += 1. * (*lambda)[getIdOffset(-1) + nbStateLabels*nbGates + g*nbFeaturesPerGate + nbFeaturesSparse];
			gates->setValue(g,nodeIndex,gate(gateSum));
		}
		
		//Then compute node feature value from gate values.
		for(int s = 0; s < nbStateLabels; s++)
		{
			for(int g = 0; g < nbGates; g++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset(seqLabel) + g + s*nbGates;
				pFeature->globalId = getIdOffset() + g + s*nbGates;
				pFeature->nodeIndex = nodeIndex;
				pFeature->nodeState = s;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = -1;
				pFeature->sequenceLabel = seqLabel;
				pFeature->value = gates->getValue(g,nodeIndex);
				pFeature->featureTypeId = GATE_NODE_FEATURE_ID;
			}
		}
	}
}

void GateNodeFeatures::getAllFeatures(featureVector& listFeatures, Model* m,
									   int nbRawFeatures)
{
	int idNode = 0;
	int nbStateLabels = m->getNumberOfStates();
	feature* pFeature;

	for(int n = -windowSize; n <= windowSize; n++)
	{
		//Node features.
		for(int s = 0; s < nbStateLabels; s++)
		{
			for(int g = 0; g < nbGates; g++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset() + g + s*nbGates;
				pFeature->globalId = getIdOffset() + g + s*nbGates;
				pFeature->nodeIndex = GATE_NODE_FEATURE_ID;
				pFeature->nodeState = s;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = -1;
				pFeature->sequenceLabel = -1;
				pFeature->value = g;
			}
		}
		
		//Gate features.
		for(int g = 0; g < nbGates; g++)
		{
			for (int f = 0; f < nbFeaturesPerGate; f++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset() + nbStateLabels*nbGates + g*nbFeaturesPerGate+ f;
				pFeature->globalId = getIdOffset() + nbStateLabels*nbGates + g*nbFeaturesPerGate+ f;
				pFeature->nodeIndex = GATE_NODE_FEATURE_ID;
				pFeature->nodeState = -1;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = -1;
				pFeature->sequenceLabel = -1;
				pFeature->value = f;
			}
		}
	}
}

void GateNodeFeatures::init(const DataSet& dataset, const Model& m)
{
	FeatureType::init(dataset,m);
	if(dataset.size() > 0)
	{
		int nbStates = m.getNumberOfStates();
		int nbSeqLabels = m.getNumberOfSequenceLabels();
		int nbFeaturesPerStates = 0;
		if((*dataset.begin())->getPrecomputedFeatures() != NULL)
			nbFeaturesPerStates += (*dataset.begin())->getPrecomputedFeatures()->getHeight();
		if((*dataset.begin())->getPrecomputedFeaturesSparse() != NULL)
			nbFeaturesPerStates += (int)(*dataset.begin())->getPrecomputedFeaturesSparse()->getHeight(); // Modified by Congkai
		int windowRange = 1 + 2*windowSize;
		
		//Total number of features needed by each gate, +1 for bias.
		nbFeaturesPerGate = nbFeaturesPerStates * windowRange + 1;
		
		//This number represents the number of weights needed. 
		//A call to getFeatures will only return nbStates * nbGates features.
		nbFeatures = nbStates * nbGates + (nbFeaturesPerStates * windowRange + 1) * nbGates;
		for(int i = 0; i < nbSeqLabels; i++)
			nbFeaturesPerLabel[i] = nbFeatures;
	}
}

bool GateNodeFeatures::isEdgeFeatureType()
{
	return false;
}

double GateNodeFeatures::gate(double sum)
{
	return 1.0/(1.0+exp(-sum));
}

void GateNodeFeatures::getPreGateFeatures(featureVector& listFeatures,
									DataSequence* X, Model* m, int nodeIndex,
									int prevNodeIndex, int seqLabel)
{
	int idNode = 0;
	int nbStateLabels = m->getNumberOfStates();
	int nbFeaturesDense = 0;
	
	dVector* lambda = m->getWeights();
	//dMatrix* gates = X->getGateMatrix();

	if(X->getPrecomputedFeatures() != NULL && prevNodeIndex == -1)
	{
		dMatrix * preFeatures = X->getPrecomputedFeatures();
		nbFeaturesDense = preFeatures->getHeight();
		int nbNodes = preFeatures->getWidth();
		
		feature* pFeature;
		for(int g = 0; g < nbGates; g++)
		{
			idNode = 0;
			for(int n = nodeIndex - windowSize; n <= nodeIndex + windowSize; n++)
			{
				if(n >= 0 && n < nbNodes)
				{
					for(int f = 0; f < nbFeaturesDense; f++)
					{
						pFeature = listFeatures.addElement();
						pFeature->id = getIdOffset(seqLabel) + nbStateLabels*nbGates + g*nbFeaturesPerGate + idNode*nbFeaturesDense + f;
						pFeature->globalId = getIdOffset() + nbStateLabels*nbGates + g*nbFeaturesPerGate + idNode*nbFeaturesDense + f;
						pFeature->nodeIndex = nodeIndex;
						pFeature->nodeState = g; //Not very clean, perhaps could find a better way to do this?
						pFeature->prevNodeIndex = -1;
						pFeature->prevNodeState = -1;
						pFeature->sequenceLabel = seqLabel;
						pFeature->value = preFeatures->getValue(f,n);
					}
				}
				idNode++;
			}
			//Gate bias.
			pFeature = listFeatures.addElement();
			pFeature->id = getIdOffset(seqLabel) + nbStateLabels*nbGates + g*nbFeaturesPerGate + nbFeaturesDense;
			pFeature->globalId = getIdOffset() + nbStateLabels*nbGates + g*nbFeaturesPerGate + idNode*nbFeaturesDense;
			pFeature->nodeIndex = nodeIndex;
			pFeature->nodeState = g; //Not very clean, perhaps could find a better way to do this?
			pFeature->prevNodeIndex = -1;
			pFeature->prevNodeState = -1;
			pFeature->sequenceLabel = seqLabel;
			pFeature->value = 1;
		}
	}

	// Load Sparse raw features
	int offsetOfPreFeatures = idNode*nbStateLabels*nbFeaturesDense;
	if(X->getPrecomputedFeaturesSparse() != NULL && prevNodeIndex == -1)
	{
		dMatrixSparse * preFeaturesSparse = X->getPrecomputedFeaturesSparse();
		int nbFeaturesSparse = (int)preFeaturesSparse->getHeight();
		int nbNodes = (int)preFeaturesSparse->getWidth();
		feature* pFeature;

		for(int g = 0; g < nbGates; g++)
		{
			idNode = 0;
			for(int n = nodeIndex - windowSize; n <= nodeIndex + windowSize; n++)
			{
				if(n >= 0 && n < nbNodes)
				{
					int irIndex = (int)preFeaturesSparse->getJc()->getValue(n);
					size_t numElementsInCol = preFeaturesSparse->getJc()->getValue(n+1) - irIndex;
					for(unsigned int i = 0; i < numElementsInCol; i++)
					{
						int f = (int)preFeaturesSparse->getIr()->getValue(irIndex + i);// feature ID
						
						pFeature = listFeatures.addElement();
						pFeature->id = getIdOffset(seqLabel) + nbStateLabels*nbGates + g*nbFeaturesPerGate + idNode*nbFeaturesDense + f;
						pFeature->globalId = getIdOffset() + nbStateLabels*nbGates + g*nbFeaturesPerGate + idNode*nbFeaturesDense + f;
						pFeature->nodeIndex = nodeIndex;
						pFeature->nodeState = g; //Not very clean, perhaps could find a better way to do this?
						pFeature->prevNodeIndex = -1;
						pFeature->prevNodeState = -1;
						pFeature->sequenceLabel = seqLabel;
						
						pFeature->value = preFeaturesSparse->getPr()->getValue(irIndex + i);
					}
				}
				idNode++;
			}
			//Gate bias.
			pFeature = listFeatures.addElement();
			pFeature->id = getIdOffset(seqLabel) + nbStateLabels*nbGates + g*nbFeaturesPerGate + nbFeaturesDense;
			pFeature->globalId = getIdOffset() + nbStateLabels*nbGates + g*nbFeaturesPerGate + idNode*nbFeaturesDense;
			pFeature->nodeIndex = nodeIndex;
			pFeature->nodeState = g; //Not very clean, perhaps could find a better way to do this?
			pFeature->prevNodeIndex = -1;
			pFeature->prevNodeState = -1;
			pFeature->sequenceLabel = seqLabel;
			pFeature->value = 1;
		}
	}
}