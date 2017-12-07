//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Trainer Component
//
//	April 5, 2006

#ifndef __TOOLBOX_H
#define __TOOLBOX_H

//Standard Template Library includes
// ...

//hCRF Library includes

#include "RawFeatures.h"
#include "FeaturesOne.h"
#include "RawFeaturesSquare.h"
#include "WindowRawFeatures.h"
#include "GaussianWindowRawFeatures.h"
#ifndef _PUBLIC
#include "MultiviewFeatures.h"
#include "BackwardWindowRawFeatures.h"
#include "SharedFeatures.h"
#include "EdgeObservationFeatures.h"
#include "GateNodeFeatures.h"
#include "StartFeatures.h"
#endif
#include "EdgeFeatures.h"
#include "LabelEdgeFeatures.h"
#include "dataset.h"
#include "model.h"
#include "optimizer.h"
#include "gradient.h"
#include "evaluator.h"
#include "inferenceengine.h"
#include "hcrfExcep.h"
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef int PORT_NUMBER;

enum {
   INIT_ZERO,
   INIT_CONSTANT,
   INIT_RANDOM,
   INIT_MEAN,
   INIT_RANDOM_MEAN_STDDEV,
   INIT_GAUSSIAN,
   INIT_RANDOM_GAUSSIAN,
   INIT_RANDOM_GAUSSIAN2,
   INIT_PREDEFINED,
   INIT_PERCEPTRON //
};

enum{
   OPTIMIZER_CG,
   OPTIMIZER_BFGS,
   OPTIMIZER_ASA,
   OPTIMIZER_OWLQN,
   OPTIMIZER_LBFGS,
   OPTIMIZER_HMMPERCEPTRON, // Discriminative training for HMM
   OPTIMIZER_NRBM // Non-convex Regularized Bundle Method (Max-Margin)
};

enum GradientType {
	GRADIENT_CRF,
	GRADIENT_CNF,
	GRADIENT_FD,
	GRADIENT_HCRF,
	GRADIENT_HCNF,
	GRADIENT_HMM_PERCEPTRON,
	GRADIENT_LDCNF,
	GRADIENT_LDCRF,
	GRADIENT_SHARED_LDCRF
};

//Toolbox, used to make predictions, must be trained before anything.
class Toolbox
{
  public:
   Toolbox();
   Toolbox(const Toolbox&);
   Toolbox& operator=(const Toolbox&){
       throw std::logic_error("Toolbox should not be copied");
   };
   virtual ~Toolbox();
   virtual void train(DataSet& X, bool bInitWeights = true);
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL) = 0;

   virtual void validate(DataSet& dataTrain, DataSet& dataValidate,
                         double& optimalRegularisation,
                         char* filenameStats = NULL);   
   
   virtual void load(char* filenameModel, char* filenameFeatures);
   virtual void save(char* filenameModel, char* filenameFeatures);
   virtual double computeError(DataSet& X);

   double getRegularizationL1();
   double getRegularizationL2();
   void setRegularizationL1(double regFactorL1,
                            eFeatureTypes typeFeature = allTypes);
   void setRegularizationL2(double regFactorL2,
                            eFeatureTypes typeFeature = allTypes);

   int getMaxNbIteration();
   int getWeightInitType();
   void setMaxNbIteration(int maxit);
   void setWeightInitType(int initType);

   void setRandomSeed(long seed);
   long getRandomSeed();

   void setInitWeights(const dVector& w);
   dVector& getInitWeights();

   void setWeights(const dVector& w);

   int getDebugLevel();
   void setDebugLevel(int newDebugLevel);

   featureVector* getAllFeatures(DataSet &X);
   Model* getModel();
   FeatureGenerator* getFeatureGenerator();
   Optimizer* getOptimizer();
   
   // Return last number of iterations run by optimizer.
   int getLastNbIterations() {if(pOptimizer) return pOptimizer->getLastNbIterations();}

   void setRangeWeights(double minRange, double maxRange);
   void setMinRangeWeights(double minRange);
   void setMaxRangeWeights(double maxRange);
   double getMinRangeWeights();
   double getMaxRangeWeights();

	//To be set before training. Will add a feature function of given ID to the feature generator.
	//This can be used to drastically modify the behavior of a CRF, for example by using RawFeaturesSquare
	//instead of just RawFeatures. When train is called, the toolbox will check for important feature
	//types missing and will fill the gaps with default feature types. For example, a regular CRF
	//will need at least one node feature type and one edge feature type, with default types being
	//WindowRawFeatures and EdgeFeatures.
	void addFeatureFunction(int featureFunctionID, int iParam1 = 0, int iParam2 = 0);

	//Will change the inference engine used for evaluation of likelihoods by the toolbox.
	void setInferenceEngine(InferenceEngine* engine);

	//To be set before training. Will change the optimizer type to use during training.
	void setOptimizer(int optimizerType);
	void setGradient(int gradientType);

	//Between the creation of the toolbox and the call to this function, users
	//are allowed to set their own InferenceEngine, Optimizer and FeatureFunctions.
	//This function checks to see if the user put up any of these things on his own,
	//and if not will put the default ones in place.
	virtual void initToolbox();

   void initWeights(DataSet &X);

   // To be able to set the number of thread using the toolbox
   // (useful from python). To change the default scheduling, one
   // must use the environment variable OMP_SCHEDULE
   void set_num_threads(int);

   virtual void initModel(DataSet &X) = 0;
   
   virtual void openPort(PORT_NUMBER portNumber, int winSize, int bufferLength) {};
   virtual void closePort(PORT_NUMBER portNumber) {};
   virtual bool insertOneFrame(PORT_NUMBER portNumber, const dVector* const features,
	   dVector* prob) {return false;}; // 0 means no value is returned.
	
  protected:

   virtual void initWeightsRandom();
   virtual void initWeightsFromMean(DataSet &X);
   virtual void initWeightsRandomFromMeanAndStd(DataSet &X);
   virtual void initWeightsGaussian(DataSet &X);

   virtual void initWeightsConstant(double value);  

   virtual void initWeightsRandomGaussian();
   virtual void initWeightsRandomGaussian2();

   virtual void initWeightsPerceptron(DataSet& X);
   virtual void calculateGlobalMean(DataSet &X, dVector&mean);
   virtual void calculateGlobalMeanAndStd(DataSet &X, dVector& mean,
                                          dVector& stdDev);

   int weightInitType;
   dVector initW;
   double minRangeWeights;
   double maxRangeWeights;
   Optimizer* pOptimizer;
   Gradient* pGradient;
   Evaluator* pEvaluator;
   Model* pModel;
   InferenceEngine* pInferenceEngine;
   FeatureGenerator* pFeatureGenerator;
   long seed;
};

class ToolboxCRF: public Toolbox
{
public:
   ToolboxCRF();
   virtual ~ToolboxCRF();

	virtual void initToolbox();

	//Real-time CRF functions
	// If port is already opened, then its content will be cleared
	virtual void openPort(PORT_NUMBER portNumber, int winSize, int bufferLength);
	virtual void closePort(PORT_NUMBER portNumber);
	virtual bool insertOneFrame(PORT_NUMBER portNumber, const dVector* const features,
		dVector* prob); // 0 means no value is returned.

	//Testing functions
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);

protected:
   virtual void initModel(DataSet &X);
	
private:
	//Real-time stuff.
	std::map<PORT_NUMBER, DataSequenceRealtime*> portNumberMap;

};

class ToolboxHCRF: public Toolbox
{
  public:
   ToolboxHCRF();

	//MATLAB will call this constructor.
	ToolboxHCRF(int nbHiddenStates);

   virtual ~ToolboxHCRF();
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);

	virtual void initToolbox();
   virtual void initModel(DataSet &X);

  protected:

  //private: {-KGB}
   int numberOfHiddenStates;
};

class ToolboxLDCRF : public Toolbox
{
  public:
   ToolboxLDCRF();
	ToolboxLDCRF(int nbHiddenStatesPerLabel);

   virtual ~ToolboxLDCRF();

	virtual void initToolbox();

	//Testing.
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);

	//Real-time LDCRF functions
	// If port is already opened, then it's content will be cleared
	virtual void openPort(PORT_NUMBER portNumber, int winSize, int bufferLength);
	virtual void closePort(PORT_NUMBER portNumber);
	virtual bool insertOneFrame(PORT_NUMBER portNumber, const dVector* const features,
		dVector* prob); // 0 means no value is returned.
   virtual void initModel(DataSet &X);

  protected:

   int numberOfHiddenStatesPerLabel;

private:
	//For real-time LDCRF.
	std::map<PORT_NUMBER, DataSequenceRealtime*> portNumberMap;

};

class ToolboxGHCRF: public ToolboxHCRF
{
  public:
   ToolboxGHCRF();

	ToolboxGHCRF(int nbHiddenStates);
	
	virtual void initToolbox();

   //ToolboxGHCRF(int nbHiddenStates, int opt, int windowSize = 0);

   virtual ~ToolboxGHCRF();

  protected:
   //virtual void init(int nbHiddenStates, int opt, int windowSize);
};

class ToolboxSharedLDCRF: public ToolboxLDCRF
{
  public:
   ToolboxSharedLDCRF();

	ToolboxSharedLDCRF(int nbHiddenStates);

	virtual void initToolbox();

   //ToolboxSharedLDCRF(int nbHiddenStates, int opt, int windowSize = 0);
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);

  protected:
   int numberOfHiddenStates;

   virtual void initModel(DataSet &X);
};

class ToolboxHMMPerceptron: public Toolbox
{
  public:
   ToolboxHMMPerceptron();

   //ToolboxHMMPerceptron(int opt, int windowSize = 0);
   virtual ~ToolboxHMMPerceptron();
   virtual void train(DataSet &X, bool bInitWeights);
   virtual double test(DataSet& X, char* filenameOutput = NULL,
                       char* filenameStats = NULL);   

	virtual void initToolbox();

  protected:
   virtual void initModel(DataSet &X);

   InferenceEnginePerceptron* pInferenceEnginePerceptron;
   GradientPerceptron* pGradientPerceptron;
};

class ToolboxLVPERCEPTRON: Toolbox
{
   public:
	ToolboxLVPERCEPTRON();

	ToolboxLVPERCEPTRON(int nbHiddenStatesPerLabel);

	//ToolboxLVPERCEPTRON(int nbHiddenStatesPerLabel,int opt, int windowSize = 0);
	virtual ~ToolboxLVPERCEPTRON();
	virtual double test(DataSet& X, char* filenameOutput = NULL, 
						char* filenameStats = NULL);
  
	virtual void initToolbox();

   protected:

	virtual void initModel(DataSet &X);
	int numberOfHiddenStatesPerLabel;
};



// Multi-View HCRF
class ToolboxMVHCRF: public Toolbox
{
public:
	ToolboxMVHCRF();

	ToolboxMVHCRF(
		eGraphTypes graphType,  
		int nbViews, 
		std::vector<int> nbHiddenStates, 
		std::vector<std::vector<int> > rawFeatureIndex);
	virtual ~ToolboxMVHCRF(); 
	
	virtual double test(DataSet& X, char* filenameOutput = NULL, char* filenameStats = NULL);
	virtual void initToolbox();

protected:
	virtual void initModel(DataSet& X);
	int m_nbViews; 
	int* m_nbHiddenStatesMultiView;
	std::vector<std::vector<int> > m_rawFeatureIndex;
	eGraphTypes m_graphType;
};

class ToolboxMVLDCRF: public Toolbox
{
public:
	ToolboxMVLDCRF();
	ToolboxMVLDCRF(
		eGraphTypes graphType,
		int nbViews,
		std::vector<int> nbHiddenStates,
		std::vector<std::vector<int> > rawFeatureIndex);
	virtual ~ToolboxMVLDCRF();

	virtual double test(DataSet& X, char* filenameOutput = NULL, char* filenameStats = NULL);
	virtual void initToolbox();

protected:
   virtual void initModel(DataSet &X);
	int m_nbViews; 
	int* m_nbHiddenStatesMultiView;
	std::vector<std::vector<int> > m_rawFeatureIndex;
	eGraphTypes m_graphType;
};

#endif
