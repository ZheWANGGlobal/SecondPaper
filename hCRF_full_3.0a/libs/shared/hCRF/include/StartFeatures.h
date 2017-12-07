//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// dummy start edge features
//
// July 25, 2011
// Julien-Charles Lévesque

#ifndef START_FEATURES_H
#define START_FEATURES_H

#include "featuregenerator.h"

//Class StartFeatures: A set of dummy features to be used in position
//0 of a given test sequence. Allows to learn probabilities for starting
//labels.
class StartFeatures : public FeatureType
{
  public:
   StartFeatures();
   void init(const DataSet& dataset, const Model& m);
   void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m,
                    int nodeIndex, int prevNodeIndex, int seqLabel = -1);
   void computeFeatureMask(iMatrix& matFeautureMask, const Model& m);
   bool isEdgeFeatureType();
   void getAllFeatures(featureVector& listFeatures, Model* m,
                       int nbRawFeatures);
};

#endif
