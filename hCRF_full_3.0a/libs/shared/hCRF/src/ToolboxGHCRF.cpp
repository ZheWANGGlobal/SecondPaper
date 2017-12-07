#include "toolbox.h"
#include "optimizer.h"

ToolboxGHCRF::ToolboxGHCRF():ToolboxHCRF()
{
}

ToolboxGHCRF::ToolboxGHCRF(int nbHiddenStates)
	: ToolboxHCRF(nbHiddenStates)
{
	setWeightInitType(INIT_GAUSSIAN);
}

/*ToolboxGHCRF::ToolboxGHCRF(int nbHiddenStates, int opt, int windowSize):ToolboxHCRF()
{
	init(nbHiddenStates,opt, windowSize);
	setWeightInitType(INIT_GAUSSIAN);
}*/

ToolboxGHCRF::~ToolboxGHCRF()
{

}

// We dont use WindowSize
void ToolboxGHCRF::initToolbox()
{
	pModel->setNumberOfStates(numberOfHiddenStates);

	//Not touching features for GHCRF toolbox.
	pFeatureGenerator = new FeatureGenerator;
	pFeatureGenerator->addFeature(new EdgeFeatures());
	pFeatureGenerator->addFeature(new LabelEdgeFeatures());

	pFeatureGenerator->addFeature(new RawFeatures());
	pFeatureGenerator->addFeature(new FeaturesOne());
	pFeatureGenerator->addFeature(new RawFeaturesSquare());

	//Inference. It is important to use the setInferenceEngine function. Otherwise,
	//if someone set a custom gradient it might break.
	if(!pInferenceEngine)
		setInferenceEngine(new InferenceEngineFB());

	//Optimizer
	if(!pOptimizer)
		pOptimizer = new OptimizerUncOptim();
		
	if(!pGradient)
		pGradient = new GradientHCRF (pInferenceEngine, pFeatureGenerator);
	pEvaluator = new EvaluatorHCRF (pInferenceEngine, pFeatureGenerator);
}

