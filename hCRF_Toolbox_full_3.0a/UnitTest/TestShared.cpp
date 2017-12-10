#include "hCRF.h"
#include "gtest/gtest.h"
#include <time.h>

/**
This file contains the test used for the development of the shared Hidden states
**/

class SharedTest : public ::testing::Test {
protected:
	virtual void SetUp(){
		dataTrain.load("data/dataTrain.csv", "data/labelsTrain.csv");
		dataTest.load("data/dataTest.csv", "data/labelsTest.csv");
		pModel = new Model();
		pModelShared = new Model();
		//Init the model
		nbr_labels = dataTest.searchNumberOfStates();
		nbrHiddenState = 2 * nbr_labels;
		pModel->setNumberOfStateLabels(nbr_labels);
		pModel->setNumberOfStates(nbrHiddenState);
		pModelShared->setNumberOfStateLabels(nbr_labels);
		pModelShared->setNumberOfStates(nbrHiddenState);
		FG = FeatureGenerator();
		pInferenceEngine = new InferenceEngineFB();
		sharedInference = new InferenceEngineDC();
	}

	virtual void TearDown()
	{
		if (pModel)
			delete pModel;
		if (pModelShared)
			delete pModelShared;
		if (pInferenceEngine)
			delete pInferenceEngine;
		if (sharedInference)
			delete sharedInference;
	};


	virtual void SetupLDCRF()
	{
		FG.clearFeatureList();
		FG.addFeature(new WindowRawFeatures(0));
		FG.addFeature(new EdgeFeatures());
		FG.initFeatures(dataTrain, *pModel);
		pModel->setStateMatType(STATES_BASED_ON_LABELS);
	};

	virtual void SetupSharedLDCRF()
	{
		FG.clearFeatureList();
		pModelShared->setStateMatType(ALLSTATES);
		pModelShared->setAdjacencyMatType(DANGLING_CHAIN);
		pModel->setStateMatType(STATES_BASED_ON_LABELS);
		pModel->setAdjacencyMatType(CHAIN);
		// Initialize feature generator
		FG.addFeature(new WindowRawFeatures(0));
		FG.addFeature(new EdgeFeatures());
		FG.addFeature(new SharedFeatures());
		FG.initFeatures(dataTrain, *pModelShared);
	}


	virtual void SetupWeights(int size, bool LDCRF, unsigned int seed)
	{
		/*
		  This function is used to create different type of weight vectors to be
		  used for testing the gradient computation. In particular we want to be
		  able to test that LDCRF and SharedLDCRF agree on particular special
		  case.
		  int size = Number of weight to generate
		  bool LDCRF = If true, the hidden state are not shared (only one value
		               not zero per column
		  unsigned int seed = seed for the random generator
		*/
		w.create(size);
		srand(seed);
		double widthRangeWeight = fabs(1.0 - (-1.0));
		for(int i = 0; i < w.getLength(); i++)
			w[i] = ( (double)rand() / (double)RAND_MAX) * widthRangeWeight - 1;

		int dimObs = dataTrain.getNumberofRawFeatures();
		if (LDCRF) {
			int start_of_HY = nbrHiddenState*(nbrHiddenState+dimObs);
			for (int i = start_of_HY; i<start_of_HY+(nbrHiddenState * nbr_labels) ; i++){
				int label = (i-start_of_HY) % nbr_labels;
				int hidden = (i-start_of_HY)/nbr_labels;
				if (label == hidden/2)
					w[i] = 50;
				else
					w[i]=-50;
			}
		}
	}

	Model* pModelShared;
	Model* pModel;
	InferenceEngine* pInferenceEngine;
	InferenceEngine* sharedInference;
	FeatureGenerator FG;
	int nbrHiddenState;
	int nbr_labels;
	DataSet dataTrain;
	DataSet dataTest;
	dVector w;
};

namespace{
	TEST_F(SharedTest, Features)
	{
		// Test that the default parameter are set correctly
		ToolboxSharedLDCRF toolbox = ToolboxSharedLDCRF(nbrHiddenState);
		toolbox.setOptimizer(OPTIMIZER_LBFGS);
		toolbox.train(dataTrain);
		toolbox.test(dataTest);
	}

	TEST_F(SharedTest, OnlyShared)
	{
		SetupSharedLDCRF();
		SetupWeights(FG.getNumberOfFeatures(), false, 32);
	}

	TEST_F(SharedTest, ZeroWeigth)
	{
		SetupSharedLDCRF();
		w = dVector(FG.getNumberOfFeatures(), COLVECTOR, 0);
		pModelShared->setWeights(w);
		EvaluatorSharedLDCRF evaluator = EvaluatorSharedLDCRF(sharedInference,
															  &FG);
		GradientSharedLDCRF gradient_test = GradientSharedLDCRF(sharedInference,
																&FG);
		GradientFD gradient_truth = GradientFD(sharedInference,
											   &FG, &evaluator);
		dVector grad_numerical = dVector();
		gradient_truth.computeGradient(grad_numerical, pModelShared, (&dataTrain));
		dVector grad_alg = dVector();
		gradient_test.computeGradient(grad_alg, pModelShared, (&dataTrain));
		
		for (int i = 0; i<w.getLength(); i++)
		{
			EXPECT_NEAR(grad_alg[i], grad_numerical[i], 1e-5)
				<<" at features_id "<<i;
		}
		double value = exp(-evaluator.computeError(dataTrain.at(4), pModelShared));
		EXPECT_NEAR(value, pow((1/3.0), 4), 1e-5);
	}

	TEST_F(SharedTest, FullGradient)
	{
		SetupSharedLDCRF();
		SetupWeights(FG.getNumberOfFeatures(), false, 0);
		 w[0] = 1;  w[1] = 0;  w[2] = 0;  
		 w[3] = 0;  w[4] = 0;  w[5] = 0;
		 w[6] = 0;  w[7] = 0;  w[8] = 0;  
		 w[9] = 0; w[10] = 0; w[11] = 0;
  		w[12] = 0; w[13] = 0; w[14] = 0; 
		w[15] = 0; w[16] = 0; w[17] = 0;
		for(int i = 0; i<36;i++){
		   w[36+i] = (i%6 == i/6)? 1:0;
		   }
		pModel->setWeights(w);
		pModelShared->setWeights(w);
		EvaluatorSharedLDCRF evaluator = EvaluatorSharedLDCRF(sharedInference,
															  &FG);
		GradientSharedLDCRF gradient_test = GradientSharedLDCRF(sharedInference,
																&FG);
		GradientFD Gradient_truth = GradientFD(sharedInference,
											   &FG, &evaluator);
		dVector grad_numerical = dVector();
		Gradient_truth.computeGradient(grad_numerical, pModelShared, 
									   (&dataTrain)->at(2));
		dVector grad_alg = dVector();
		gradient_test.computeGradient(grad_alg, pModelShared, (&dataTrain)->at(2));
		for (int i = 0; i<w.getLength(); i++){
		   EXPECT_NEAR(grad_alg[i], grad_numerical[i], 1e-5);
		}
	}

	TEST_F(SharedTest, gradient)
	{
		/* a FEW SETUP */
		SetupSharedLDCRF();
		SetupWeights(FG.getNumberOfFeatures(), true, 0);
		pModelShared->setWeights(w);
		/* Compute value for SharedLDCRF */
		EvaluatorSharedLDCRF evaluator = EvaluatorSharedLDCRF(sharedInference,
															  &FG);
		GradientSharedLDCRF gradient_test = GradientSharedLDCRF(sharedInference,
																&FG);
		GradientFD Gradient_truth = GradientFD(pInferenceEngine, &FG,
											   &evaluator);
		dVector grad_num = dVector();
		double value = evaluator.computeError(&dataTrain, pModelShared);
		Gradient_truth.computeGradient(grad_num, pModelShared, (&dataTrain));
		dVector grad_alg = dVector();
		gradient_test.computeGradient(grad_alg, pModelShared, (&dataTrain));

		/* Compute for LDCRF */
		SetupLDCRF();
		dVector w2 = dVector(FG.getNumberOfFeatures());;
		for (int i = 0; i<w2.getLength(); i++){
			w2[i] = w[i];
		}
		pModel->setWeights(w2);
		GradientLDCRF gradient = GradientLDCRF(pInferenceEngine, &FG);
		EvaluatorLDCRF evaluator_ldcrf = EvaluatorLDCRF(pInferenceEngine, &FG);
		dVector grad_ldcrf = dVector();
		double value_ldcrf = evaluator_ldcrf.computeError( &dataTrain, pModel);
		gradient.computeGradient(grad_ldcrf, pModel, (&dataTrain));
		for (int i = 0; i<w2.getLength(); i++){
			EXPECT_NEAR(grad_num[i], grad_ldcrf[i], 1e-5) << 
				"at features id "<< i;
			EXPECT_NEAR(grad_alg[i], grad_num[i], 1e-5) << 
				"at features id "<< i;
		}
		EXPECT_NEAR(value, value_ldcrf, 1e-8);
	}
	
}
