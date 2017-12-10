#include "BackwardWindowRawFeatures.h"

using namespace std;


BackwardWindowRawFeatures::BackwardWindowRawFeatures(int winSize)
: FeatureType()
, WindowSize(winSize)
{
	strFeatureTypeName = "Backward Window Raw Feature Type";
	featureTypeId = BACKWARD_WINDOW_RAW_FEATURE_ID;
	basicFeatureType = NODE_FEATURE;
}


void BackwardWindowRawFeatures::getFeatures(featureVector& listFeatures,
									DataSequence* X, Model* m, int nodeIndex,
									int prevNodeIndex, int seqLabel)
{
	int idNode = 0;
	int nbStateLabels = m->getNumberOfStates();
	int nbFeaturesDense = 0;

	//Check if we are using Realtime DataSequence...
	DataSequenceRealtime* data = dynamic_cast<DataSequenceRealtime*>(X);
	bool realtime = false;
	if (data)
		realtime = true;

	if(X->getPrecomputedFeatures() != NULL && prevNodeIndex == -1)
	{		
		dMatrix * preFeatures = X->getPrecomputedFeatures();
		nbFeaturesDense = preFeatures->getHeight();
		int nbNodes = preFeatures->getWidth();
		feature* pFeature;

		for(int n = nodeIndex - WindowSize; n <= nodeIndex; n++)
		{
			int pos = n;
			if (realtime)
				pos = (n+nbNodes) % nbNodes;
			
			if(pos >= 0 && pos < nbNodes)
			{					
				for(int s = 0; s < nbStateLabels; s++)
				{
					for(int f = 0; f < nbFeaturesDense; f++)
					{
						pFeature = listFeatures.addElement();
						pFeature->id = getIdOffset(seqLabel) + f + s*nbFeaturesDense + idNode*nbStateLabels*nbFeaturesDense;
						pFeature->globalId = getIdOffset() + f + s*nbFeaturesDense + idNode*nbStateLabels*nbFeaturesDense;
						pFeature->nodeIndex = nodeIndex;
						pFeature->nodeState = s;
						pFeature->prevNodeIndex = -1;
						pFeature->prevNodeState = -1;
						pFeature->sequenceLabel = seqLabel;
						pFeature->value = preFeatures->getValue(f,pos); // TODO: Optimize
					}										
				}
			}
			idNode++;
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
		idNode = 0;
		
		for(int n = nodeIndex - WindowSize; n <= nodeIndex; n++)
		{
			int pos = n;
			if (realtime)
				pos = (n+nbNodes) % nbNodes;

			if(pos >= 0 && pos < nbNodes)
			{
				int irIndex = (int)preFeaturesSparse->getJc()->getValue(pos);
				size_t numElementsInCol = preFeaturesSparse->getJc()->getValue(pos+1) - irIndex;
				for(int s = 0; s < nbStateLabels; s++)
				{
					for(unsigned int i = 0; i < numElementsInCol; i++)
					{						
						int f = (int)preFeaturesSparse->getIr()->getValue(irIndex + i);// feature ID 					
						pFeature = listFeatures.addElement();					
						pFeature->id = getIdOffset(seqLabel) + f + s*nbFeaturesSparse + idNode*nbStateLabels*nbFeaturesSparse + offsetOfPreFeatures;						
						pFeature->globalId = getIdOffset() + f + s*nbFeaturesSparse + idNode*nbStateLabels*nbFeaturesSparse + offsetOfPreFeatures;
						
						pFeature->nodeIndex = nodeIndex;
						pFeature->nodeState = s;
						pFeature->prevNodeIndex = -1;
						pFeature->prevNodeState = -1;
						pFeature->sequenceLabel = seqLabel;
						
						pFeature->value = preFeaturesSparse->getPr()->getValue(irIndex + i);
					}					
				}
			}
			idNode++;
		}
	}
}

void BackwardWindowRawFeatures::getAllFeatures(featureVector& listFeatures, Model* m,
									   int nbRawFeatures)
{
	int idNode = 0;
	int nbStateLabels = m->getNumberOfStates();
	feature* pFeature;

	for(int n = -WindowSize; n <= 0; n++)
	{
		for(int s = 0; s < nbStateLabels; s++)
		{
			for(int f = 0; f < nbRawFeatures; f++)
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset() + f + s*nbRawFeatures + idNode*nbStateLabels*nbRawFeatures;
				pFeature->globalId = getIdOffset() + f + s*nbRawFeatures + idNode*nbStateLabels*nbRawFeatures;
				pFeature->nodeIndex = BACKWARD_WINDOW_RAW_FEATURE_ID;
				pFeature->nodeState = s;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = -1;
				pFeature->sequenceLabel = -1;
				pFeature->value = f; 
			}
		}
		idNode++;
	}
}

void BackwardWindowRawFeatures::init(const DataSet& dataset, const Model& m)
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
		int windowRange = 1 + WindowSize;

		nbFeatures = nbStates * nbFeaturesPerStates * windowRange;
		for(int i = 0; i < nbSeqLabels; i++)
			nbFeaturesPerLabel[i] = nbFeatures;
	}
}

bool BackwardWindowRawFeatures::isEdgeFeatureType()
{
	return false;
}
