#include "optimizer.h"

#ifdef USEOWL
OptimizerOWL::~OptimizerOWL()
{
}

OptimizerOWL::OptimizerOWL():
	currentModel(NULL), currentDataset(NULL),
	currentEvaluator(NULL), currentGradient(NULL), vecGradient()
{
}

void OptimizerOWL::optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad)
{
	currentModel = m;
	currentDataset = X;
	currentEvaluator = eval;
	currentGradient= grad;
	double regweight = 0.0;
	double tol = 1e-2;
	int memsize = 10;
	size_t size = currentModel->getWeights()->getLength();
	DblVec init, ans(size);
	vecGradient.create((int)size);
	// Set the initial weights
	init.assign( currentModel->getWeights()->get(), 
				 currentModel->getWeights()->get() + size );
	// Optimize
	if (m->getDebugLevel() >= 1) {
		opt.SetQuiet(false);
	} else {
		opt.SetQuiet(true);
	}
	opt.Minimize(*this, init, ans, regweight, tol, memsize);
	// Save the optimal weights
	memcpy(vecGradient.get(),&ans[0],size*sizeof(double));
	currentModel->setWeights(vecGradient);
	dVector tmpWeights = *(currentModel->getWeights());
	tmpWeights.transpose();
	tmpWeights.multiply(*currentModel->getWeights());
	lastNormGradient = tmpWeights[0];
}

// Compute gradient and evaluate error function
double OptimizerOWL::Eval(const DblVec& input, DblVec& gradient)
{
	dVector dgrad((int)input.size());
	// Copy current weights
	memcpy(vecGradient.get(),&input[0],input.size()*sizeof(double));
	currentModel->setWeights(vecGradient);
	// Evaluate error function
	if(currentModel->getDebugLevel() >= 1) {
		std::cout<<"New Iteration"<<std::endl;
		if (currentModel->getDebugLevel() >= 3){
			std::cout<<"x = "<<currentModel->getWeights()<<std::endl;
		}
		if(currentModel->getDebugLevel() >= 2) {
			std::cout << "Compute error..." << std::endl;
		}
	}
	double f = currentEvaluator->computeError(currentDataset, currentModel);
	//Compute gradient
	if(currentModel->getDebugLevel() >= 2)
		std::cout << "Compute gradient..." << std::endl;
	currentGradient->computeGradient(dgrad, currentModel,currentDataset);
	if (currentModel->getDebugLevel() >= 3){
		std::cout<<"g = "<<dgrad<<std::endl;
	}
	//Copy gradient
	gradient.assign(dgrad.get(),dgrad.get()+input.size());
	return f;
}
#endif
