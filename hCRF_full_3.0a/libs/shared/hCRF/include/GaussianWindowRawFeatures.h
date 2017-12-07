//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Gaussian Window Raw Features
//
// Based on Song et al., 
// "Multi-Signal Gesture Recognition Using Temporal Smoothing HCRF", FG 2011	
//
// Yale Song (yalesong@csail.mit.edu)
// July, 2011


#ifndef GAUSSIAN_WINDOW_RAW_FEATURES_H
#define GAUSSIAN_WINDOW_RAW_FEATURES_H

#include "featuregenerator.h"

class GaussianWindowRawFeatures: public FeatureType
{
public:
	GaussianWindowRawFeatures(int windowSize = 0);
	~GaussianWindowRawFeatures();

	virtual void init(const DataSet& dataset, const Model& m);

	virtual void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M,
		int nodeIndex, int prevNodeIndex, int seqLabel = -1);
	void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);

	virtual bool isEdgeFeatureType();

private:
	int windowSize;
	double* weights;
};

#endif



