//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Backward window raw features
//
//	February 20, 2006

#ifndef WINDOW_RAW_FEATURES_REALTIME_H
#define WINDOW_RAW_FEATURES_REALTIME_H

#include "featuregenerator.h"

//BackwardWindowRawFeatures
//
// These features are a special case of window raw features. They never look forward,
//only backward, meaning that if the window has a size of 2, it will look at y(t-2),
//y(t-1) and y(t). It will not look at y(t+1) or y(t+2).
class BackwardWindowRawFeatures : public FeatureType
{
public:
	BackwardWindowRawFeatures(int windowSize = 0);

	virtual void init(const DataSet& dataset, const Model& m);
	virtual void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, 
					int nodeIndex, int prevNodeIndex, int seqLabel = -1);
	virtual bool isEdgeFeatureType();

	void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);
private:
	int WindowSize;
};

#endif 
