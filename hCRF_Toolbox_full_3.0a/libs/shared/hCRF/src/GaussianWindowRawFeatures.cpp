//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Gaussian Window Raw Features
//
// Based on Song et al., 
// "Multi-Signal Gesture Recognition Using Temporal Smoothing HCRF", FG 2011	
//
// TODO:
//  1. Implement loading sparse raw features in getFeatures()
//
// Yale Song (yalesong@csail.mit.edu)
// July, 2011

#include "GaussianWindowRawFeatures.h"

GaussianWindowRawFeatures::GaussianWindowRawFeatures(int ws): FeatureType()
{
	strFeatureTypeName = "Gaussian Window Raw Feature Type";
	featureTypeId = GAUSSIAN_WINDOW_RAW_FEATURE_ID;
	basicFeatureType = NODE_FEATURE;

	windowSize = 1 + 2*ws;
	
	// Compute a normalized Gaussian kernel. 
	weights = new double[windowSize];
	double sum=0;
	if( windowSize == 1 )
	{
		weights[0] = 1.0;
	}
	else 
	{
		double alpha = 2.5;
		int N = windowSize-1;
		for( int i=0-N/2; i<=N-N/2; i++ )
		{
			double x = alpha * (i / (0.5*N));
			weights[i+N/2] = exp(-0.5*x*x);
			sum += weights[i+N/2];
		}
		for( int i=0; i<windowSize; i++ )
		{
			weights[i] /= sum;
		}
	}
}

GaussianWindowRawFeatures::~GaussianWindowRawFeatures()
{
	if( weights ) {
		delete [] weights;
		weights = 0;
	}
}

void GaussianWindowRawFeatures::getFeatures(featureVector& listFeatures, 
	DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
	int nbStateLabels = m->getNumberOfStates();
	int nbFeaturesDense = 0;

	if( X->getPrecomputedFeatures()!=NULL && prevNodeIndex == -1 )
	{
		dMatrix *preFeatures = X->getPrecomputedFeatures();
		nbFeaturesDense = preFeatures->getHeight();
		int nbNodes = preFeatures->getWidth();
		feature* pFeature;

		for( int s=0; s<nbStateLabels; s++ ) 
		{
			for( int f=0; f<nbFeaturesDense; f++ )
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset(seqLabel) + f + s*nbFeaturesDense;
				pFeature->globalId = getIdOffset() + f + s*nbFeaturesDense;
				pFeature->nodeIndex = nodeIndex;
				pFeature->nodeState = s;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = -1;
				pFeature->sequenceLabel = seqLabel;
				pFeature->value = 0; 

				for( int w=0; w<windowSize; w++ )
				{
					int idx = nodeIndex - (w-(int)(windowSize/2));
					if( idx<0 || idx>nbNodes-1 ) continue;
					pFeature->value += weights[w] * preFeatures->getValue(f,idx);
				}
			}
		}
	}

	// TODO_YALE: load sparse raw features
}

void GaussianWindowRawFeatures::getAllFeatures(
	featureVector& listFeatures, Model* m, int nbRawFeatures)
{
	int nbStateLabels = m->getNumberOfStates();
	feature* pFeature;

	for( int s=0; s<nbStateLabels; s++ )
	{
		for( int f=0; f<nbRawFeatures; f++ )
		{
			pFeature = listFeatures.addElement();
			pFeature->id = getIdOffset() + f + s*nbRawFeatures;
			pFeature->globalId = getIdOffset() + f + s*nbRawFeatures;
			pFeature->nodeIndex = featureTypeId;
			pFeature->nodeState = s;
			pFeature->prevNodeIndex = -1;
			pFeature->prevNodeState = -1;
			pFeature->sequenceLabel = -1;
			pFeature->value = f;  
		}
	}
}

void GaussianWindowRawFeatures::init(const DataSet& dataset, const Model& m)
{
	FeatureType::init(dataset, m);
	if( dataset.size() > 0 )
	{
		int nbStates = m.getNumberOfStates();
		int nbSeqLabels = m.getNumberOfSequenceLabels();
		int nbFeaturesPerState = dataset.at(0)->getPrecomputedFeatures()->getHeight();

		nbFeatures = nbStates * nbFeaturesPerState;
		for( int i=0; i<nbSeqLabels; i++ )
		{
			nbFeaturesPerLabel[i] = nbFeatures;
		}
	}
}

bool GaussianWindowRawFeatures::isEdgeFeatureType()
{
	return false;
}