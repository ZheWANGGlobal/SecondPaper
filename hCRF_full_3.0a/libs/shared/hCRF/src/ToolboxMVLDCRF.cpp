//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Multi-View LDCRF toolbox
//
// Yale Song (yalesong@csail.mit.edu)
// October, 2011


#include "toolbox.h"

ToolboxMVLDCRF::ToolboxMVLDCRF(): Toolbox() 
{
}


ToolboxMVLDCRF::ToolboxMVLDCRF(eGraphTypes gt, int nv, 
	std::vector<int> nhs, std::vector<std::vector<int> > rfi)
: m_graphType(gt), m_nbViews(nv), m_rawFeatureIndex(rfi), Toolbox()
{ 
	m_nbHiddenStatesMultiView = new int[m_nbViews]; // For LDCRF, this is per label
	for( int i=0; i<m_nbViews; i++ )
		m_nbHiddenStatesMultiView[i] = nhs[i];   

	// This is necessary to prevent Toolbox::addFeatureFunction() from discarding
	// same typed features with different view index
	pModel->setNumberOfViews(m_nbViews); 
	pModel->setAdjacencyMatType(m_graphType);
	pModel->setRawFeatureIndexMV(m_rawFeatureIndex);
	pModel->setNumberOfStatesMV(m_nbHiddenStatesMultiView);
}


ToolboxMVLDCRF::~ToolboxMVLDCRF()
{
	if( m_nbHiddenStatesMultiView ) {
		delete [] m_nbHiddenStatesMultiView;
		m_nbHiddenStatesMultiView = 0;
	}
}


void ToolboxMVLDCRF::initToolbox()
{
	// Add features
	if (!pFeatureGenerator)
		pFeatureGenerator = new FeatureGenerator();

	if (!pFeatureGenerator->getFeatureByBasicType(NODE_FEATURE)) {
		for(int i=0; i<m_nbViews; i++) {
			addFeatureFunction(MV_GAUSSIAN_WINDOW_RAW_FEATURE_ID, i, 0);
		}		
	} 

	// Edge features (pairwise potentials)
	for( int i=0; i<m_nbViews; i++ ) 
	{
		// add MV_EDGE_WITHIN_VIEW (c==d, s+1=t)
		addFeatureFunction(MV_EDGE_FEATURE_ID, i, i);

		// add MV_EDGE_BETWEEN_VIEW (c!=d, s==t)
		if( m_graphType==MV_GRAPH_LINKED || m_graphType==MV_GRAPH_LINKED_COUPLED )
			for( int j=i+1; j<m_nbViews; j++ )
				addFeatureFunction(MV_EDGE_FEATURE_ID, i, j); 

		// add MV_EDGE_CROSS_VIEW (c!=d, s+1=t)
		if( m_graphType==MV_GRAPH_COUPLED || m_graphType==MV_GRAPH_LINKED_COUPLED ) { 
			for( int j=0; j<m_nbViews; j++ ) {
				if( i==j ) continue;
				addFeatureFunction(MV_EDGE_FEATURE_ID, i, j);
			}
		} 
	} 

	if( !pInferenceEngine )
		pInferenceEngine = new InferenceEngineLoopyBP();
	if( !pGradient )
		pGradient = new GradientMVLDCRF(pInferenceEngine, pFeatureGenerator);
	if( !pEvaluator )
		pEvaluator = new EvaluatorMVLDCRF(pInferenceEngine, pFeatureGenerator);

	Toolbox::initToolbox(); 
	
}
 

void ToolboxMVLDCRF::initModel(DataSet &X)
{   
	int nbStateLabels = X.searchNumberOfStates();
	int* nbStates = new int[m_nbViews];
	for(int i=0; i<m_nbViews; i++)
		nbStates[i] = m_nbHiddenStatesMultiView[i] * nbStateLabels;
	pModel->setNumberOfStatesMV(nbStates);	
	delete[] nbStates; nbStates = 0;
 
	pModel->setNumberOfStateLabels(nbStateLabels);
	pModel->setStateMatType(STATES_BASED_ON_LABELS);

	pFeatureGenerator->initFeatures(X, *pModel); 
}

double ToolboxMVLDCRF::test(DataSet& X, char* filenameOutput, char* filenameStats)
{ 
	// For Matlab interface
	initModel(X);

	double returnedF1value = 0.0;
	std::ofstream* fileOutput = NULL;
	if(filenameOutput)
	{
		fileOutput = new std::ofstream(filenameOutput);
		if (!fileOutput->is_open())
		{
			delete fileOutput;
			fileOutput = NULL;
		}
	}
	std::ostream* fileStats = NULL;
	if(filenameStats)
	{
		fileStats = new std::ofstream(filenameStats, std::ios_base::out | std::ios_base::app);
		if (!((std::ofstream*)fileStats)->is_open())
		{
			delete fileStats;
			fileStats = NULL;
		}
	}
	if(fileStats == NULL && pModel->getDebugLevel() >= 1)
		fileStats = &std::cout; 

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
  

	for(it = X.begin(); it != X.end(); it++) 
	{
		// Compute detected label
		dMatrix* matProbabilities = new dMatrix;
		iVector* vecLabels = new iVector;
		pEvaluator->computeStateLabels(
				*it,pModel,vecLabels,matProbabilities,getModel()->isMaxMargin());
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
		(*fileStats) << std::endl << "Calculations per sample:" << std::endl;
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
		(*fileStats) << std::endl << "Calculations per sequence:" << std::endl;
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

