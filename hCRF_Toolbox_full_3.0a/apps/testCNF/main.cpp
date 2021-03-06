#include <stdio.h>
#include "hCRF.h"

#ifdef WIN32
#include <conio.h>
#endif

int main(int argc, char **argv)
{
	int nbGates = 40;
	int windowSize = 1;
	int nbHiddenStates = 2;
	
	//Start by creating a regular CRF or LDCRF toolbox.
	ToolboxLDCRF toolbox(nbHiddenStates);

	//Start features are optional but provide a nice performance boost.
	toolbox.addFeatureFunction(START_FEATURE_ID);

	//Edge features are also optional (they will be added automatically by the toolbox
	//if they are found missing at the moment of training).
	toolbox.addFeatureFunction(EDGE_FEATURE_ID);

	//Instead of using standard window raw features, CNF and LDCNF use gate node features,
	//which contain the neural network's gates and also the node features (links between
	//the gates and the states).
	toolbox.addFeatureFunction(GATE_NODE_FEATURE_ID,nbGates,windowSize);

	//Use whatever optimizer you like.
	toolbox.setOptimizer(OPTIMIZER_LBFGS);

	//To train a CNF or LDCNF, you need to set the gradient to the CNF version, because otherwise the 
	//gates will not be weighted properly.
	toolbox.setGradient(GRADIENT_LDCNF);
	
	DataSet X;
	X.load("../../../learning_toolbox/data/taskar_split/training_data_w_bias.csv","../../../learning_toolbox/data/taskar_split/training_labels.csv");
	
	toolbox.setMaxNbIteration(100);
	toolbox.setRegularizationL2(0.5);
	toolbox.setDebugLevel(1);
	
	toolbox.train(X,true);
	toolbox.test(X,"train","train_stats");
	
	DataSet testX;
	testX.load("../../../learning_toolbox/data/taskar_split/val_data_w_bias.csv","../../../learning_toolbox/data/taskar_split/val_labels.csv");
	toolbox.test(testX,"test","test_stats");

	toolbox.save("ldcnf_model","ldcnf_features");
	
	system("pause");
	printf("Done.\n");
	return 0;
}
