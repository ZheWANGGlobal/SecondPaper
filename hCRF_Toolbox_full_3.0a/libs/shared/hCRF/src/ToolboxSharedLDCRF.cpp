#include "toolbox.h"

ToolboxSharedLDCRF::ToolboxSharedLDCRF():ToolboxLDCRF()
{
}

ToolboxSharedLDCRF::ToolboxSharedLDCRF(int nbHidStates)
{
	numberOfHiddenStates = nbHidStates;
}

void ToolboxSharedLDCRF::initToolbox()
{
	Toolbox::initToolbox();

	// We need the right inference engine
	if(pInferenceEngine!=NULL) {
		delete pInferenceEngine;
	}
	pInferenceEngine = new InferenceEngineDC();

	// The order of the features in the feature generator is as follow: 
	//  - WindowsRawFeatures (nbrStates * dimObs) Features between obs
	//       and hidden 
	//  - EdgeFeatures (nbrStates * nbrStates) Features for the link
	//       between hidden states.
    //  - SharedFeatures (nbrLabels * nbrStates features) weights between
	//       labels and states 
	//JCL: Currently, there is no way of checking features ordering,
	//although by default (without manually adding feature functions) their ordering will be correct.
	if (!pFeatureGenerator->getFeatureById(LATENT_LABEL_FEATURE_ID))
		pFeatureGenerator->addFeature(new SharedFeatures());

	pEvaluator = new EvaluatorSharedLDCRF (pInferenceEngine, pFeatureGenerator);
	if(!pGradient)
		pGradient = new GradientSharedLDCRF(pInferenceEngine, pFeatureGenerator);
}

void ToolboxSharedLDCRF::initModel(DataSet &X)
{
	int nbr_labels = X.searchNumberOfStates();	
	iMatrix SharedMatrix;
	pModel->setNumberOfStateLabels(nbr_labels);
	pModel->setNumberOfStates(numberOfHiddenStates);
	pModel->setNumberOfSequenceLabels(0);
	pModel->setAdjacencyMatType(DANGLING_CHAIN);
	pModel->setStateMatType(ALLSTATES);
	// Initialize feature generator
	pFeatureGenerator->initFeatures(X,*pModel);
}

double ToolboxSharedLDCRF::test(DataSet& X, char* filenameOutput, 
								char* filenameStats)
{
	double returnedF1value = 0.0;
	std::ofstream* fileOutput = NULL;
	if(filenameOutput) {
		fileOutput = new std::ofstream(filenameOutput);
		if (!fileOutput->is_open()) {
			delete fileOutput;
			fileOutput = NULL;
		}
	}
	std::ostream* fileStats = NULL;
	if(filenameStats){
		fileStats = new std::ofstream(filenameStats, std::ios_base::out | 
									  std::ios_base::app);
		if (!((std::ofstream*)fileStats)->is_open()) {
			delete fileStats;
			fileStats = NULL;
		}
	}
	if(fileStats == NULL && pModel->getDebugLevel() >= 1){
		fileStats = &std::cout;
	}
	DataSet::iterator it;
	int nbStateLabels = pModel->getNumberOfStateLabels();
	iVector seqTruePos(nbStateLabels);
	iVector seqTotalPos(nbStateLabels);
	iVector seqTotalPosDetected(nbStateLabels);
	iVector truePos(nbStateLabels);
	iVector totalPos(nbStateLabels);
	iVector totalPosDetected(nbStateLabels);
	iVector tokenPerLabel(nbStateLabels);
	iVector tokenPerLabelDetected(nbStateLabels);
	// We update the model by moving the weight of sharedfeature
	// into LabelProbability and removing them from the
	// weight vector
	for(it = X.begin(); it != X.end(); it++) 
	{
		//  Compute detected labels
		dMatrix* matProbabilities = new dMatrix;
		iVector* vecLabels = new iVector;
		pEvaluator->computeStateLabels(*it,pModel,vecLabels, matProbabilities);
		(*it)->setEstimatedStateLabels(vecLabels);
		(*it)->setEstimatedProbabilitiesPerStates(matProbabilities);

		// optionally writes results in file
		if( fileOutput)
		{
			for(int i = 0; i < (*it)->length(); i++)
			{
				(*fileOutput) << (*it)->getStateLabels(i) << "\t" << vecLabels->getValue(i);
				for(int l = 0; l < nbStateLabels; l++)
					(*fileOutput) << "\t" << matProbabilities->getValue(l,i);			
				(*fileOutput) << std::endl;
			}
		}

		//Count state labels for the sequence
		tokenPerLabel.set(0);
		tokenPerLabelDetected.set(0);
		for(int i = 0; i < (*it)->length(); i++)
		{
		    tokenPerLabel[(*it)->getStateLabels(i)]++;
			tokenPerLabelDetected[vecLabels->getValue(i)]++;

			totalPos[(*it)->getStateLabels(i)]++;
			totalPosDetected[vecLabels->getValue(i)]++;

			if(vecLabels->getValue(i) == (*it)->getStateLabels(i))
				truePos[vecLabels->getValue(i)]++;
		}
		//Find max label for the sequence
		int maxLabel = 0;
		int maxLabelDetected = 0;
		for(int j = 1 ; j < nbStateLabels ; j++) 
		{
		    if(tokenPerLabel[maxLabel] < tokenPerLabel[j])
		    	maxLabel = j;
		    if(tokenPerLabelDetected[maxLabelDetected] < tokenPerLabelDetected[j])
		    	maxLabelDetected = j;
		}
		(*it)->setEstimatedSequenceLabel(maxLabelDetected);
		// Update total of positive detections
		seqTotalPos[maxLabel]++;
		seqTotalPosDetected[maxLabelDetected]++;
		if( maxLabel == maxLabelDetected)
			seqTruePos[maxLabel]++;
	}
	// Print results
	if(fileStats)
	{
		(*fileStats) << std::endl << "Calculations per samples:" << std::endl;
		(*fileStats) << "Label\tTrue+\tMarked+\tDetect+\tPrec.\tRecall\tF1" << std::endl;
	}
	double prec,recall;
	int SumTruePos = 0, SumTotalPos = 0, SumTotalPosDetected = 0;
	for(int i=0 ; i<nbStateLabels ; i++) 
	{
		SumTruePos += truePos[i]; SumTotalPos += totalPos[i]; SumTotalPosDetected += totalPosDetected[i];
		prec=(totalPos[i]==0)?0:((double)truePos[i])*100.0/((double)totalPos[i]);
		recall=(totalPosDetected[i]==0)?0:((double)truePos[i])*100.0/((double)totalPosDetected[i]);
		if(fileStats)
			(*fileStats) << i << ":\t" << truePos[i] << "\t" << totalPos[i] << "\t" << totalPosDetected[i] << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
	}
	prec=(SumTotalPos==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPos);
	recall=(SumTotalPosDetected==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPosDetected);
	if(fileStats)
	{
		(*fileStats) << "-----------------------------------------------------------------------" << std::endl;
		(*fileStats) << "Ov:\t" << SumTruePos << "\t" << SumTotalPos << "\t" << SumTotalPosDetected << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
	}
	returnedF1value = 2*prec*recall/(prec+recall);

	if(fileStats)
	{
		(*fileStats) << std::endl << "Calculations per sequences:" << std::endl;
		(*fileStats) << "Label\tTrue+\tMarked+\tDetect+\tPrec.\tRecall\tF1" << std::endl;
		SumTruePos = 0, SumTotalPos = 0, SumTotalPosDetected = 0;
		for(int i=0 ; i<nbStateLabels ; i++) 
		{
			SumTruePos += seqTruePos[i]; SumTotalPos += seqTotalPos[i]; SumTotalPosDetected += seqTotalPosDetected[i];
			prec=(seqTotalPos[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPos[i]))/1000;
			recall=(seqTotalPosDetected[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPosDetected[i]))/1000;
			(*fileStats) << i << ":\t" << seqTruePos[i] << "\t" << seqTotalPos[i] << "\t" << seqTotalPosDetected[i] << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
		}
		prec=(SumTotalPos==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPos);
		recall=(SumTotalPosDetected==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPosDetected);
		(*fileStats) << "-----------------------------------------------------------------------" << std::endl;
		(*fileStats) << "Ov:\t" << SumTruePos << "\t" << SumTotalPos << "\t" << SumTotalPosDetected << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
	}

	if( fileOutput )
	{
		fileOutput->close();
		delete fileOutput;
	}
	if(fileStats != &std::cout && fileStats != NULL)
	{
		((std::ofstream*)fileStats)->close();
		delete fileStats;
	}
	return returnedF1value;
}
