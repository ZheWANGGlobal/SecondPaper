//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Gradient
// Component
//
//	Aug, 2010

#include "gradient.h"
#include <exception>

GradientPerceptron::GradientPerceptron(InferenceEnginePerceptron* infEngine, FeatureGenerator* featureGen)
	:pInfEngine(infEngine), pFeatureGen(featureGen)
{}

GradientPerceptron::GradientPerceptron(const GradientPerceptron& other)
	:pInfEngine(other.pInfEngine), pFeatureGen(other.pFeatureGen)
{}

GradientPerceptron& GradientPerceptron::operator=(const GradientPerceptron& other)
{
	pInfEngine = other.pInfEngine;
	pFeatureGen = other.pFeatureGen;
	return *this;
}

// Batch Update
double GradientPerceptron::computeGradient(dVector& vecGradient, Model* m, DataSet* X, bool bComputeMaxMargin)
{
	if( bComputeMaxMargin )
		throw HcrfNotImplemented("GradientPerceptron for max-margin is not implemented");

	double converge = 1; // converge = 1 when training finished
	
	//Check the size of vecGradient
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);
	else
		vecGradient.set(0);

	dVector tmpVecGradient(nbFeatures);	
    for(unsigned int i = 0; i< X->size(); i++){
		tmpVecGradient.set(0.0);
		double updated = computeGradient(tmpVecGradient, m, X->at(i));
		if(updated == 1)
		{
			tmpVecGradient.multiply(X->at(i)->getWeightSequence());
			vecGradient.add(tmpVecGradient);
			converge = 0;
		}
	}		
	return converge;
}

