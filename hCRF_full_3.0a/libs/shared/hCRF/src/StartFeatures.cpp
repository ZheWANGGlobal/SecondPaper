#include "StartFeatures.h"

using namespace std;


StartFeatures::StartFeatures():FeatureType()
{
	strFeatureTypeName = "Start Feature Type";
	featureTypeId = START_FEATURE_ID;
	//Special because it cannot be used as a replacement to NODE_FEATUREs.
	basicFeatureType = SPECIAL_FEATURE;
}

void StartFeatures::getFeatures(featureVector& listFeatures, DataSequence* X, 
							   Model* m, int nodeIndex, int prevNodeIndex, 
							   int seqLabel)
{	
	//Will behave like a node feature.
	if(prevNodeIndex == -1 && nodeIndex == 0)
	{
		feature* pFeature;
		int nbStateLabels = m->getNumberOfStates();
		for(int s = 0; s < nbStateLabels; s++)
		{
			pFeature = listFeatures.addElement();
			pFeature->id = getIdOffset(seqLabel) + s;
			pFeature->globalId = getIdOffset() + s + seqLabel*nbStateLabels;
			pFeature->nodeIndex = 0;
			pFeature->nodeState = s;
			pFeature->prevNodeIndex = -1;
			pFeature->prevNodeState = -1;
			pFeature->sequenceLabel = seqLabel;
			pFeature->value = 1.0f;
		}
	}
}

void StartFeatures::getAllFeatures(featureVector& listFeatures, Model* m, 
								  int)
/* We dont need the number of raw features as the number of edge feature is
 * independant from the size of the windows
 */
{
	int nbStateLabels = m->getNumberOfStates();
	int nbSeqLabels = m->getNumberOfSequenceLabels();
	feature* pFeature;
	if(nbSeqLabels == 0)
		nbSeqLabels = 1;
	for(int seqLabel = 0; seqLabel < nbSeqLabels;seqLabel++)
	{
		for(int s = 0; s < nbStateLabels; s++)
		{
			pFeature = listFeatures.addElement();
			pFeature->id = getIdOffset() + s + seqLabel*nbStateLabels;
			pFeature->globalId = getIdOffset() + s + seqLabel*nbStateLabels;
			pFeature->nodeIndex = featureTypeId;
			pFeature->nodeState = s;
			pFeature->prevNodeIndex = -1;
			pFeature->prevNodeState = -1;
			pFeature->sequenceLabel = seqLabel;
			pFeature->value = 1.0f;
		}
	}
}


void StartFeatures::init(const DataSet& dataset, const Model& m)
{
	FeatureType::init(dataset,m);
	int nbStateLabels = m.getNumberOfStates();
	int nbSeqLabels = m.getNumberOfSequenceLabels();

	if(nbSeqLabels == 0)
		nbFeatures = nbStateLabels;
	else
	{
		nbFeatures = nbStateLabels*nbSeqLabels;
		for(int i = 0; i < nbSeqLabels; i++)
			nbFeaturesPerLabel[i] = nbStateLabels;
	}
}

void StartFeatures::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
{
	int nbLabels = m.getNumberOfSequenceLabels();
	int firstOffset = idOffset;

	for(int j = 0; j < nbLabels; j++)
	{
		int lastOffset = firstOffset + nbFeaturesPerLabel[j];
	
		for(int i = firstOffset; i < lastOffset; i++)
			matFeautureMask(i,j) = 1;

		firstOffset += nbFeaturesPerLabel[j];
	}
}

bool StartFeatures::isEdgeFeatureType()
{
	return false;
}
