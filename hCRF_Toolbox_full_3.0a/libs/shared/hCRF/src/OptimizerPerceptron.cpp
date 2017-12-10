
#include "optimizer.h"

OptimizerPerceptron::OptimizerPerceptron(typeOptimizer defaultOptimizer)
{
}

OptimizerPerceptron::~OptimizerPerceptron()
{
}

void OptimizerPerceptron::optimize(Model* m, DataSet* X,Evaluator* eval, GradientPerceptron* grad)
{
	currentModel = m;
	currentDataset = X;
	//currentEvaluator = eval;
	currentGradient= grad;
	int nbWeights = currentModel->getWeights()->getLength();
	vecGradient.create(nbWeights);

	for(int i =0; i< maxit; i++)
	{
		double converge = currentGradient->computeGradient(vecGradient, currentModel, currentDataset);				
		if(converge ==1)
			break;
		dVector* weights = currentModel->getWeights();
		weights->add(vecGradient);
		//currentModel->setWeights(vecGradient);
	}
	/* Report the result. */
	if(currentModel->getDebugLevel() >= 1)
	{
		//std::cout << "L-BFGS optimization terminated with status code = " << ret <<std::endl;
		//std::cout << "  fx = " << fx << std::endl;
	}
}