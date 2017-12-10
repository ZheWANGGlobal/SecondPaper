//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Shared edge features (Features between labels and hidden states
//
//	hsalamin, January 28, 2010

#ifndef SHARED_FEATURES_H
#define SHARED_FEATURES_H

#include "featuregenerator.h"


class SharedFeatures : public FeatureType
{
public:
	SharedFeatures ();
	void init(const DataSet& dataset, const Model& m);
	void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, 
                   int nodeIndex, int prevNodeIndex, int seqLabel = -1);
	bool isEdgeFeatureType(){return true;};
	void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
	void getAllFeatures(featureVector& listFeatures, Model* m, int nbrRawFeature);
};

#endif 
