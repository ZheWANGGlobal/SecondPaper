//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Gate node features
//
//	Julien-Charles Lévesque
//	July 14th 2011

#ifndef GATE_NODE_FEATURES_H
#define GATE_NODE_FEATURES_H

#include "featuregenerator.h"

/*
Gate node features.

A layer made of nbGates gates will be inserted between the raw observations
and the classical node features. The learning of these features need to be
done by the CNF or LDCNF gradients, depending on which case you find yourself 
in. 

TODO: Add bias automatically to each observation instead of having to add
them directly on the dataset...

*/
class GateNodeFeatures : public FeatureType
{
public:
	GateNodeFeatures(int nbGates, int windowSize = 0);

	virtual void init(const DataSet& dataset, const Model& m);
	virtual void getFeatures(featureVector& listFeatures, DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel = -1);
	virtual bool isEdgeFeatureType();

	void getAllFeatures(featureVector& listFeatures, Model* m, int nbRawFeatures);

	int getNbFeaturesPerGate() {return nbFeaturesPerGate;}
	int getNbGates() {return nbGates;}

	//Function used by gradient to have direct access to raw features.
	void getPreGateFeatures(featureVector& listFeatures, DataSequence* X, Model* m, int nodeIndex, int prevNodeIndex, int seqLabel = -1);

private:
	//Gating function of the neural network. h(x) = 1/(1+exp(x))
	double gate(double sum);

	int nbFeaturesPerGate;
	int windowSize;
	int nbGates;
};

#endif 
