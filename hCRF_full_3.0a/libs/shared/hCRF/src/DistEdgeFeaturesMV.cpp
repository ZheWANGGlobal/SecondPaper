//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Distant Edge Features for Multi-View models.
//
// Encodes dependency between h^{c}_{s} and h^{d}_{t}. 
//
// Yale Song (yalesong@csail.mit.edu)
// January, 2012

#include "MultiviewFeatures.h"

// To use M_PI
#define _USE_MATH_DEFINES
#include "math.h"
#include <stdlib.h>

DistEdgeFeaturesMV::DistEdgeFeaturesMV(int v1, int v2, bool timeShare):FeatureType(),
prevViewIdx(v1), curViewIdx(v2), isTimeShare(timeShare)
{
	strFeatureTypeName = "Multiview Edge Feature Type";
	featureTypeId = MV_DISTEDGE_FEATURE_ID;
	basicFeatureType = EDGE_FEATURE;
}

void DistEdgeFeaturesMV::getFeatures(featureVector& listFeatures, 
	DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel)
{
	if( prevNodeIndex == -1 ) return;
	
	int T = X->length();
	int nodeView = nodeIndex/T;
	int prevNodeView = prevNodeIndex/T;
	if( prevViewIdx!=prevNodeView || curViewIdx!=nodeView ) return; 
	 
	int nodeTime = nodeIndex % T;
	int prevNodeTime = prevNodeIndex % T;
	if( isTimeShare && prevNodeTime!=nodeTime ) return;
	if( !isTimeShare && prevNodeTime==nodeTime ) return;

	int absDiff = abs(prevNodeTime-nodeTime);
	double var = 1.0;
	double val = exp(-0.5*absDiff*absDiff/var)/sqrt(2*M_PI*var);
	//double val = ( absDiff < 2 ) ? 1 : 1 / sqrt((double)absDiff);
		
	feature* pFeature;
	int nbStateLabels1 = m->getNumberOfStatesMV(prevViewIdx);
	int nbStateLabels2 = m->getNumberOfStatesMV(curViewIdx);

	for( int s1=0; s1<nbStateLabels1; s1++ )
	{
		for( int s2=0; s2<nbStateLabels2; s2++ )
		{
			pFeature = listFeatures.addElement();
			pFeature->id = getIdOffset(seqLabel) + s2 + s1*nbStateLabels2;
			pFeature->globalId = getIdOffset() + s2 + s1*nbStateLabels2 + seqLabel*nbStateLabels1*nbStateLabels2;
			pFeature->nodeView = nodeView;
			pFeature->nodeIndex = nodeIndex;
			pFeature->nodeState = s2;
			pFeature->prevNodeView = prevNodeView;
			pFeature->prevNodeIndex = prevNodeIndex;
			pFeature->prevNodeState = s1;
			pFeature->sequenceLabel = seqLabel;
			pFeature->value = val;
		}
	}
}

void DistEdgeFeaturesMV::getAllFeatures(featureVector& listFeatures, Model* m, int)
{
	int nbStateLabels1 = m->getNumberOfStatesMV(prevViewIdx);
	int nbStateLabels2 = m->getNumberOfStatesMV(curViewIdx);
	
	int nbSeqLabels = m->getNumberOfSequenceLabels();	
	if( nbSeqLabels==0 ) nbSeqLabels = 1; 
	
	feature* pFeature;	
	for( int seqLabel=0; seqLabel<nbSeqLabels; seqLabel++ )
	{
		for(int s1=0; s1<nbStateLabels1; s1++ )
		{
			for(int s2=0; s2<nbStateLabels2; s2++ )
			{
				pFeature = listFeatures.addElement();
				pFeature->id = getIdOffset() + s2 + s1*nbStateLabels2 + seqLabel*nbStateLabels1*nbStateLabels2;
				pFeature->globalId = getIdOffset() + s2 + s1*nbStateLabels2 + seqLabel*nbStateLabels1*nbStateLabels2;
				pFeature->nodeView = featureTypeId;
				pFeature->nodeIndex = featureTypeId;
				pFeature->nodeState = s2;
				pFeature->prevNodeView = -1;
				pFeature->prevNodeIndex = -1;
				pFeature->prevNodeState = s1;
				pFeature->sequenceLabel = seqLabel;
				pFeature->value = 1;
			}
		}
	}
}


void DistEdgeFeaturesMV::init(const DataSet& dataset, const Model& m)
{
	FeatureType::init(dataset,m);
	int nbStateLabels1 = m.getNumberOfStatesMV(prevViewIdx);
	int nbStateLabels2 = m.getNumberOfStatesMV(curViewIdx);

	int nbSeqLabels = m.getNumberOfSequenceLabels();
	if( nbSeqLabels==0 )
	{
		nbFeatures = nbStateLabels1 * nbStateLabels2;
	}
	else
	{
		nbFeatures = nbSeqLabels * nbStateLabels1 * nbStateLabels2;
		for( int i=0; i<nbSeqLabels; i++ )
			nbFeaturesPerLabel[i] = nbStateLabels1*nbStateLabels2;
	}
}

void DistEdgeFeaturesMV::computeFeatureMask(iMatrix& matFeautureMask, const Model& m)
{
	int i, y, nbLabels, firstOffset, lastOffset;

	nbLabels = m.getNumberOfSequenceLabels();
	firstOffset = idOffset;

	for(y=0; y<nbLabels; y++) {
		lastOffset = firstOffset + nbFeaturesPerLabel[y];
		for(i=firstOffset; i<lastOffset; i++)
			matFeautureMask(i,y) = 1;
		firstOffset += nbFeaturesPerLabel[y];
	}
}

bool DistEdgeFeaturesMV::isEdgeFeatureType()
{
	return true;
}


