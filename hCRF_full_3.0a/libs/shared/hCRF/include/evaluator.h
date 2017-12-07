//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Evaluator
// Component
//
//	January 30, 2006

#ifndef EVALUATOR_H
#define EVALUATOR_H

//Standard Template Library includes
#include <list>

//hCRF Library includes
#include "featuregenerator.h"
#include "inferenceengine.h"
#include "dataset.h"
#include "model.h"
#include "hcrfExcep.h"

#ifdef _OPENMP
#include <omp.h>
#endif

class Evaluator 
{
  public:
    Evaluator();
    Evaluator(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    virtual ~Evaluator();
    Evaluator(const Evaluator& other);
    Evaluator& operator=(const Evaluator& other);
    void init(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    virtual double computeError(DataSequence* X, Model* m, bool bComputeMaxMargin=false) = 0;
    virtual double computeError(DataSet* X, Model* m, bool bComputeMaxMargin=false);
    virtual void computeStateLabels(DataSequence* X, Model* m, 
                                    iVector* vecStateLabels, 
                                    dMatrix * probabilities = NULL, 
									bool bComputeMaxMargin=false);
    virtual int computeSequenceLabel(DataSequence* X, Model* m, 
                                     dMatrix * probabilities, 
									 bool bComputeMaxMargin=false);
	 void setInferenceEngine(InferenceEngine* inEngine){pInfEngine = inEngine;}
    
  protected:
    InferenceEngine* pInfEngine;
    FeatureGenerator* pFeatureGen;
    void computeLabels(Beliefs& bel, iVector* vecStateLabels,
                       dMatrix * probabilities = NULL);
    friend class OptimizerLBFGS;
};


class EvaluatorCRF:public Evaluator
{
  public:
    EvaluatorCRF();
    EvaluatorCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorCRF();
    double computeError(DataSequence* X, Model* m, bool bComputeMaxMargin=false);
    virtual double computeError(DataSet* X, Model * m, bool bComputeMaxMargin=false){
        return Evaluator::computeError(X,m,bComputeMaxMargin);
    }

};

class EvaluatorHCRF:public Evaluator
{
  public:
    EvaluatorHCRF();
    EvaluatorHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorHCRF();
    double computeError(DataSequence* X, Model* m, bool bComputeMaxMargin=false);
    virtual double computeError(DataSet* X, Model * m, bool bComputeMaxMargin=false){
        return Evaluator::computeError(X,m,bComputeMaxMargin);
    }

    int computeSequenceLabel(DataSequence* X, Model* m, 
                             dMatrix * probabilities, 
							 bool bComputeMaxMargin=false);
};

class EvaluatorLDCRF:public Evaluator
{
  public:
    EvaluatorLDCRF();
    EvaluatorLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorLDCRF();
    virtual double computeError(DataSequence* X, Model* m, bool bComputeMaxMargin=false);
    virtual double computeError(DataSet* X, Model * m, bool bComputeMaxMargin=false){
        return Evaluator::computeError(X,m,bComputeMaxMargin);
    }
    void computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, 
                            dMatrix * probabilities = NULL, 
							bool bComputeMaxMargin=false);
};

class EvaluatorLVPERCEPTRON:public Evaluator
{
public:
	EvaluatorLVPERCEPTRON();
	EvaluatorLVPERCEPTRON(InferenceEngine* infEngine, FeatureGenerator* featureGen);
	~EvaluatorLVPERCEPTRON();

	double computeError(DataSequence* X, Model* m, bool bComputeMaxMargin=false);
	double computeError(DataSet* X, Model* m, bool bComputeMaxMargin=false);

	void computeStateLabels(DataSequence* X, Model* m, 
							iVector* vecStateLabels, 
							dMatrix * probabilities = NULL, 
							bool bComputeMaxMargin=false);

};


class EvaluatorSharedLDCRF:public EvaluatorLDCRF
{
  public:
	EvaluatorSharedLDCRF(InferenceEngine* infEngine, 
                       FeatureGenerator* featureGen) 
      : EvaluatorLDCRF(infEngine, featureGen) {};   
    void computeStateLabels(DataSequence* X, Model* m, iVector* vecStateLabels, 
                            dMatrix * probabilities = NULL, 
							bool bComputeMaxMargin=false);
};


class EvaluatorDD: public Evaluator
{
public:
	EvaluatorDD(InferenceEngine* infEngine, FeatureGenerator* featureGen, 
              dVector* mu=0);
	~EvaluatorDD();
	double computeError(DataSet* X, Model* m, bool bComputeMaxMargin=false);
	double computeError(DataSequence* X, Model* m, bool bComputeMaxMargin=false);

private:
	dVector mu;
};


class EvaluatorMVHCRF:public Evaluator
{
  public:
    EvaluatorMVHCRF();
    EvaluatorMVHCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorMVHCRF();
    double computeError(DataSequence* X, Model* m, bool bComputeMaxMargin=false);
    virtual double computeError(DataSet* X, Model * m, bool bComputeMaxMargin=false){
        return Evaluator::computeError(X,m,bComputeMaxMargin);
    }

    int computeSequenceLabel(DataSequence* X, Model* m, dMatrix * probabilities, bool bComputeMaxMargin=false);
};

class EvaluatorMVLDCRF:public Evaluator
{
  public:
    EvaluatorMVLDCRF();
    EvaluatorMVLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen);
    ~EvaluatorMVLDCRF();
    double computeError(DataSequence* X, Model* m, bool bComputeMaxMargin=false);
    virtual double computeError(DataSet* X, Model * m, bool bComputeMaxMargin=false){
        return Evaluator::computeError(X,m,bComputeMaxMargin);
    }

    void computeStateLabels(
		DataSequence* X, 
		Model* m, 
		iVector* vecStateLabels, 
		 dMatrix * probabilities = NULL, 
		 bool bComputeMaxMargin=false);
};
#endif 
    