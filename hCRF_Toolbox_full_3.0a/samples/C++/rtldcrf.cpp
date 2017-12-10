#include <stdio.h>
#include "hCRF.h"

#ifdef WIN32
#include <conio.h>
#endif

/*

Example of real-time latent-dynamic conditonal random field.

Julien-Charles Levesque
8/19/2011

This sample will contain a simulation of real-time ldcrf and
some commented code as to the way to do it properly using a
real peripheral.

*/

int main(int argc, char **argv)
{
	int windowSize = 1;
	int nbHiddenStates = 2;
	
	//Start by creating a regular CRF or LDCRF toolbox.
	ToolboxLDCRF toolbox(nbHiddenStates);

	//Edge features are optional (they will be added automatically by the toolbox
	//if they are found missing at the moment of training), but lets add them
	//for the sake of knowing what is used as feature functions.
	toolbox.addFeatureFunction(EDGE_FEATURE_ID);

	//Backward window features only get information from previous observations, not
	//the coming observations. This allows us to reduce the delay.
	toolbox.addFeatureFunction(BACKWARD_WINDOW_FEATURE_ID,windowSize);

	//Use whatever optimizer you like.
	toolbox->setOptimizer(OPTIMIZER_LBFGS);

	DataSet X();
	X.load("dataTrain.csv","labelsTrain.csv");
	
	toolbox.setMaxNbIteration(100);
	toolbox.setRegularizationL2(0.5);
	toolbox.setDebugLevel(1);

	//Train using the regular inference engine, because it gives better performance.
	//Instead of training everytime, one can just load a model.
	toolbox.train(X,true);

	//Then change it to the ForwardFilter Inference Engine, which simulates RT.
	//The delay value used for simulation is the length of the buffer we would keep
	//in a real case.
	int delay = 2;
	toolbox.setInferenceEngine(new InferenceEngineFF(delay));

	bool simulate = true;
	if (simulate)
	{
		toolbox.test(X,"train","train_stats");
	}
	//Else, lets say we have a real device capturing and we want to classify based
	// on the data from that device.
	else
	{
		bool end = false;
		//2 in the case of binary classification
		dVector* prob = new dVector(2); 
		dVector* features;

		toolbox.openPort(0,windowSize,delay);
		while (!end)
		{
			//grab new frame from device, something like
			//grabNewFeatures(features);

			int ready = toolbox.insertOneFrame(0, featureVec, prob);

			if (ready)
			{
				//Means buffer got full, do something with the probabilities produced!
			}
		}
	}

	//toolbox.save("ldcrf_model","ldcrf_features");
	
	system("pause");
	printf("Done.\n");
	return 0;
}