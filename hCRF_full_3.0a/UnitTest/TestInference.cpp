#include "hCRF.h"
#include "gtest/gtest.h"


class TestInferenceDC : public ::testing::Test{
protected:
	virtual void SetUp(){
		dataTrain.load("data/dataInference.csv", "data/labelsInference.csv");
		nbrHiddenState = 2;
		pModel = new Model();
		//Init the model
		nbr_labels = dataTrain.searchNumberOfStates();
		pModel->setNumberOfStateLabels(nbr_labels);
		pModel->setNumberOfStates(nbrHiddenState * nbr_labels);
		nbrHiddenState = nbrHiddenState * nbr_labels;
		FG = FeatureGenerator();
		FG.clearFeatureList();
		pModel->setStateMatType(ALLSTATES);
		pModel->setAdjacencyMatType(DANGLING_CHAIN);
		// Initialize feature generator
		FG.addFeature(new WindowRawFeatures(0));
		FG.addFeature(new EdgeFeatures());
		FG.addFeature(new SharedFeatures());
		FG.initFeatures(dataTrain, *pModel);
		// Set were the weight_label start
		sharedInference = new InferenceEngineDC();
		w = dVector(FG.getNumberOfFeatures(), COLVECTOR, 0);
		pModel->setWeights(w);
	}

	virtual void TearDown()
	{
		if (pModel)
			delete pModel;
		if (sharedInference)
			delete sharedInference;
	};
	Model* pModel;
	InferenceEngine* sharedInference;
	FeatureGenerator FG;
 	int nbrHiddenState;
	int nbr_labels;
	DataSet dataTrain;
	dVector w;
};

namespace{
	TEST_F(TestInferenceDC, NotHYWeights)
	{
		w[12] = log(2.0);
		pModel->setWeights(w);
		Beliefs bel;
		sharedInference->computeBeliefs(bel, &FG, dataTrain.at(0), pModel, 0, 
										-1, false);
		// The following result where computed by hand
		EXPECT_NEAR(bel.belStates[0][0], 0.29411, 1e-5);
		EXPECT_NEAR(bel.belStates[0][1], 0.23529, 1e-5);
		EXPECT_NEAR(bel.belEdges[0](0, 0), 0.11764, 1e-5);
		EXPECT_NEAR(bel.belStates[2][0], 0.5, 1e-5);
		EXPECT_NEAR(bel.belStates[2][1], 0.5, 1e-5);
		EXPECT_NEAR(bel.belEdges[2](0,0), 10/68.0, 1e-5);
		EXPECT_NEAR(bel.belEdges[2](0,1), 8/68.0, 1e-5);
		for(int i= 0; i<bel.belStates[0].getLength(); i++){
			EXPECT_NEAR(bel.belStates[0][i], bel.belStates[1][i], 1e-8);
		}
	}
	TEST_F(TestInferenceDC, WithHYWeigths)
	{
		w[28] = log(2.0); //Weight between h=0 and y=0
		w[12] = log(2.0); //weight between h_i=0 and h_i+1=0
		pModel->setWeights(w);
		Beliefs bel;
		sharedInference->computeBeliefs(bel, &FG, dataTrain.at(0), pModel, 0, 
										-1, false);
		// The following result where computed by hand
		EXPECT_NEAR(bel.belStates[0][0], 0.4, 1e-5);
		EXPECT_NEAR(bel.belStates[0][1], 0.2, 1e-5);
		EXPECT_NEAR(bel.belEdges[0](0, 0), 0.20, 1e-5);
		EXPECT_NEAR(bel.belStates[2][0], 51.0/90, 1e-8);
		EXPECT_NEAR(bel.belStates[2][1], 39.0/90, 1e-8);
		for(int i= 0; i<bel.belStates[0].getLength(); i++){
			EXPECT_NEAR(bel.belStates[0][i], bel.belStates[1][i], 1e-8);
		}
	}
	TEST_F(TestInferenceDC, UseTheY)
	{
		w[28] = log(2.0); //Weight between h=0 and y=0
		w[12] = log(2.0); //weight between h_i=0 and h_i+1=0
		pModel->setWeights(w);
		Beliefs bel;
		sharedInference->computeBeliefs(bel, &FG, dataTrain.at(0), pModel, 0, 
										-1, true);
		EXPECT_NEAR(bel.belStates[0][0], 0.318181, 1e-5);
		EXPECT_NEAR(bel.belStates[0][1], 0.227272, 1e-5);
		EXPECT_NEAR(bel.belStates[1][0], 0.454545, 1e-5);
		EXPECT_NEAR(bel.belStates[1][1], 0.181818, 1e-5);
		EXPECT_NEAR(bel.belEdges[0](0, 0), 0.181818, 1e-5);
		EXPECT_NEAR(bel.belEdges[0](1, 0), 0.090909, 1e-5);
		EXPECT_NEAR(bel.belEdges[0](0, 1), 0.0454545, 1e-5);
	}

	TEST_F(TestInferenceDC, Partition)
	{
		w[28] = log(2.0); //Weight between h=0 and y=0
		w[12] = log(2.0); //weight between h_i=0 and h_i+1=0
		pModel->setWeights(w);
		Beliefs bel;
		sharedInference->computeBeliefs(bel, &FG, dataTrain.at(0), pModel, 0, 
										-1, true);
		double partY = bel.partition;
		sharedInference->computeBeliefs(bel, &FG, dataTrain.at(0), pModel, 0, 
										-1, false);
		double partTotal = bel.partition;
		EXPECT_NEAR(exp(partY), 22.0 , 1e-5);
		EXPECT_NEAR(exp(partTotal), 90.0, 1e-5);
		EXPECT_NEAR(exp(partY-partTotal), 0.2444444, 1e-5);
	}

	TEST_F(TestInferenceDC, LogIgnoreY)
	{
		w[28] = log(2.0); //Weight between h=0 and y=0
		w[12] = log(2.0); //weight between h_i=0 and h_i+1=0
		pModel->setWeights(w);
		Beliefs bel;
		sharedInference->computeBeliefs(bel, &FG, dataTrain.at(0), pModel, 0, 
										-1, false);
		Beliefs logBel;
		sharedInference->computeBeliefs(logBel, &FG, dataTrain.at(0), pModel, 0,
										   -1, false);
		for(int i = 0; i<dataTrain.at(0)->length(); i++){
			for(int j=0; j<nbrHiddenState; j++){
				EXPECT_NEAR(logBel.belStates[i][j],
							bel.belStates[i][j] , 1e-5)<<i<<","<<j;
			}
		}
		for(int i = 0; i<dataTrain.at(0)->length()-1; i++){
			for(int j=0; j<nbrHiddenState; j++){
				for (int k=0; k<nbrHiddenState; k++){
					EXPECT_NEAR(logBel.belEdges[i](j,k),
								bel.belEdges[i](j,k) , 1e-5)<<i<<","<<j<<","<<k;
				}
			}
		}
		for(int i = dataTrain.at(0)->length(); i<2*dataTrain.at(0)->length(); i++){
			for(int j=0; j<nbr_labels; j++){
				for (int k=0; k<nbrHiddenState; k++){
					EXPECT_NEAR(logBel.belEdges[i](j,k),
								bel.belEdges[i](j,k) , 1e-5)<<i<<","<<j<<","<<k;
				}
			}
		}
	}

	TEST_F(TestInferenceDC, LogWithY)
	{
		w[28] = log(2.0); //Weight between h=0 and y=0
		w[12] = log(2.0); //weight between h_i=0 and h_i+1=0
		pModel->setWeights(w);
		Beliefs bel;
		sharedInference->computeBeliefs(bel, &FG, dataTrain.at(0), pModel, 0, 
										-1, true);
		Beliefs logBel;
		sharedInference->computeBeliefs(logBel, &FG, dataTrain.at(0), pModel, 0,
										   -1, true);
		for(int i = 0; i<dataTrain.at(0)->length(); i++){
			for(int j=0; j<nbrHiddenState; j++){
				EXPECT_NEAR(logBel.belStates[i][j],
							bel.belStates[i][j] , 1e-5);
			}
		}
		for(int i = 0; i<dataTrain.at(0)->length()-1; i++){
			for(int j=0; j<nbrHiddenState; j++){
				for (int k=0; k<nbrHiddenState; k++){
					EXPECT_NEAR(logBel.belEdges[i](j,k),
								bel.belEdges[i](j,k) , 1e-5)<<i<<","<<j<<","<<k;
				}
			}
		}
		for(int i = dataTrain.at(0)->length(); i<2*dataTrain.at(0)->length(); i++){
			for(int j=0; j<nbr_labels; j++){
				for (int k=0; k<nbrHiddenState; k++){
					EXPECT_NEAR(logBel.belEdges[i](j,k),
								bel.belEdges[i](j,k) , 1e-5)<<i<<","<<j<<","<<k;
				}
			}
		}
	}


}
