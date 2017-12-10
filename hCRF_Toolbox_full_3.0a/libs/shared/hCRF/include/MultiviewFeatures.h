//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// a family of Multiview Features 
//
// Yale Song (yalesong@csail.mit.edu)
// July, 2011

#ifndef MULTIVIEW_FEATURES_H
#define MULTIVIEW_FEATURES_H

#include "featuregenerator.h"

class GaussianWindowRawFeaturesMV: public FeatureType
{
public:
	GaussianWindowRawFeaturesMV(int viewIdx, int windowSize=0);
	~GaussianWindowRawFeaturesMV();

	void init(const DataSet& dataset, const Model& m);

	void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M,
		int nodeIndex, int prevNodeIndex, int seqLabel = -1); 
	void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);

	bool isEdgeFeatureType();

private:
	int viewIdx;
	int windowSize;
	double* weights;
};


class EdgeFeaturesMV: public FeatureType
{
public:
   EdgeFeaturesMV(int prevViewIdx, int prevTimeIdx);

   void init(const DataSet& dataset, const Model& m);

   void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M, 
	   int nodeIndex, int prevNodeIndex, int seqLabel = -1);   
   void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);

   void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
   bool isEdgeFeatureType();

private:
	int prevViewIdx, curViewIdx; 
};


class DistEdgeFeaturesMV: public FeatureType
{
public:
   DistEdgeFeaturesMV(int prevViewIdx, int prevTimeIdx, bool isTimeShare);

   void init(const DataSet& dataset, const Model& m);

   void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M, 
	   int nodeIndex, int prevNodeIndex, int seqLabel = -1);   
   void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);

   void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
   bool isEdgeFeatureType();

private:
	int prevViewIdx, curViewIdx;
	bool isTimeShare;
};

class LabelEdgeFeaturesMV: public FeatureType
{
public:
	LabelEdgeFeaturesMV(int viewIdx);

	void init(const DataSet& dataset, const Model& m);

	void getFeatures(featureVector& listFeatures, DataSequence* X, Model* M, 
		int nodeIndex, int prevNodeIndex, int seqLabel = -1); 
	void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);

	void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
	bool isEdgeFeatureType();

private:
	int viewIdx;
};


#endif



