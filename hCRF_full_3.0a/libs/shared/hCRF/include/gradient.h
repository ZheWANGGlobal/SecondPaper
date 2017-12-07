//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Gradient
// Component
//
//	January 30, 2006

#ifndef GRADIENT_H
#define GRADIENT_H

//Standard Template Library includes
#include <vector>
#include <assert.h>

//hCRF Library includes
#include "dataset.h"
#include "model.h"
#include "inferenceengine.h"
#include "featuregenerator.h"
#include "evaluator.h"

#ifdef _OPENMP
#include <omp.h>
#endif

class Gradient {
  public:
    Gradient(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    Gradient(const Gradient&);
    Gradient& operator=(const Gradient&);
    virtual double computeGradient(dVector& vecGradient, Model* m, 
                                 DataSequence* X, bool bComputeMaxMargin=false) = 0;
    virtual double computeGradient(dVector& vecGradient, Model* m, 
                                 DataSet* X, bool bComputeMaxMargin=false);
    virtual ~Gradient();
	virtual void setMaxNumberThreads(int maxThreads);
	void setInferenceEngine(InferenceEngine* inEngine){pInfEngine = inEngine;}

  protected:
    InferenceEngine* pInfEngine;
    FeatureGenerator* pFeatureGen;
	featureVector *vecFeaturesMP;
	dVector *localGrads;
	int nbThreadsMP;
};

class GradientCRF : public Gradient 
{
  public:
    GradientCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    using Gradient::computeGradient;
};

class GradientHCRF : public Gradient 
{
  public:
    GradientHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    using Gradient::computeGradient;
private:
	double computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X);
	double computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X);
};

class GradientLDCRF : public Gradient 
{
  public:
    GradientLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    virtual double computeGradient(dVector& vecGradient, Model* m, DataSet* X);
};

class GradientSharedLDCRF : public Gradient
{
  public:
  GradientSharedLDCRF(InferenceEngine* infEngine, 
                        FeatureGenerator* featureGen):
    Gradient(infEngine, featureGen){};
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    using Gradient::computeGradient;
};

/*
 * Gradient for Sparse hidden dynamics conditional random fields, or Sparse Shared LDCRF.
 * The only difference between this gradient and the GradientSharedLDCRF for now will be
 * the 
 * */
/*class GradientSHDCRF : public Gradient
{
  public:
  GradientSHDCRF(InferenceEngine* infEngine, 
                        FeatureGenerator* featureGen):
    Gradient(infEngine, featureGen){};
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X);
    using Gradient::computeGradient;
};*/

class GradientDD : public Gradient 
{
  public:
    GradientDD(InferenceEngine* infEngine, FeatureGenerator* featureGen, 
               dVector* pMu = NULL);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    double computeGradient(dVector& vecGradient, Model* m, DataSet *X, bool bComputeMaxMargin=false);
    
  private:
    dVector mu;
};

class GradientFD : public Gradient
{
  public:
    GradientFD(InferenceEngine* infEngine, FeatureGenerator* featureGen, 
               Evaluator* evaluator);
    GradientFD(const GradientFD&);
    GradientFD& operator=(const GradientFD&);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    // We do the numerical deirvative directly on the sum (we do not
    // use the Gradient::computeGradient function)
    double computeGradient(dVector& vecGradient, Model* m, DataSet* X, bool bComputeMaxMargin=false);
  private:
    Evaluator* pEvaluator;
};

class GradientPerceptron {
  public:
    GradientPerceptron(InferenceEnginePerceptron* infEngine, FeatureGenerator* featureGen);
    GradientPerceptron(const GradientPerceptron&);
    GradientPerceptron& operator=(const GradientPerceptron&);
    virtual double computeGradient(dVector& vecGradient, Model* m, 
                                 DataSequence* X, bool bComputeMaxMargin=false) = 0;
    virtual double computeGradient(dVector& vecGradient, Model* m, 
                                 DataSet* X, bool bComputeMaxMargin=false);
    virtual ~GradientPerceptron(){};

  protected:
    InferenceEnginePerceptron* pInfEngine;
    FeatureGenerator* pFeatureGen;
};

class GradientHMMPerceptron : public GradientPerceptron 
{
  public:
    GradientHMMPerceptron(InferenceEnginePerceptron* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    using GradientPerceptron::computeGradient;
};

class GradientCNF : public Gradient 
{
  public:
    GradientCNF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    using Gradient::computeGradient;
};

class GradientHCNF : public Gradient
{
  public:
    GradientHCNF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    double computeGradient(dVector& vecGradient, Model* m, DataSet* X);
};

class GradientLDCNF : public Gradient 
{
  public:
    GradientLDCNF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    virtual double computeGradient(dVector& vecGradient, Model* m, DataSet* X);
};

class GradientMVHCRF: public Gradient
{
public:
    GradientMVHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    using Gradient::computeGradient;

private:
	double computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X);
	double computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X); 
};

class GradientMVLDCRF: public Gradient
{
public:
    GradientMVLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    double computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin=false);
    using Gradient::computeGradient;
private:
	double computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X);
	double computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X);
};

#endif
