//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Multi-View HCRF toolbox
//
// Yale Song (yalesong@csail.mit.edu)
// July, 2011


#include "toolbox.h"

ToolboxMVHCRF::ToolboxMVHCRF(): Toolbox() 
{
}


ToolboxMVHCRF::ToolboxMVHCRF(eGraphTypes gt, int nv, 
	std::vector<int> nhs, std::vector<std::vector<int> > rfi)
: m_graphType(gt), m_nbViews(nv), m_rawFeatureIndex(rfi), Toolbox()
{
	m_nbHiddenStatesMultiView = new int[m_nbViews];
	for( int i=0; i<m_nbViews; i++ )
		m_nbHiddenStatesMultiView[i] = nhs[i]; 
	
	// This is necessary to prevent Toolbox::addFeatureFunction() from discarding
	// same typed features with different view index
	pModel->setNumberOfViews(m_nbViews); 
	pModel->setAdjacencyMatType(m_graphType);
	pModel->setNumberOfStatesMV(m_nbHiddenStatesMultiView);
	pModel->setRawFeatureIndexMV(m_rawFeatureIndex);
}


ToolboxMVHCRF::~ToolboxMVHCRF()
{
	if( m_nbHiddenStatesMultiView ) {
		delete [] m_nbHiddenStatesMultiView;
		m_nbHiddenStatesMultiView = 0;
	}
}


void ToolboxMVHCRF::initToolbox()
{
	// Add features
	if (!pFeatureGenerator)
		pFeatureGenerator = new FeatureGenerator();

	if (!pFeatureGenerator->getFeatureByBasicType(NODE_FEATURE)) {
		for(int i=0; i<m_nbViews; i++) {
			addFeatureFunction(MV_GAUSSIAN_WINDOW_RAW_FEATURE_ID, i, 0);
		}		
	}
	if (!pFeatureGenerator->getFeatureByBasicType(LABEL_EDGE_FEATURE)) {
		for(int i=0; i<m_nbViews; i++) {
			addFeatureFunction(MV_LABEL_EDGE_FEATURE_ID, i);
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

	Toolbox::initToolbox(); 
	
	if( !pInferenceEngine )
		pInferenceEngine = new InferenceEngineLoopyBP();
	if( !pGradient )
		pGradient = new GradientMVHCRF(pInferenceEngine, pFeatureGenerator);
	if( !pEvaluator )
		pEvaluator = new EvaluatorMVHCRF(pInferenceEngine, pFeatureGenerator);
}
 

void ToolboxMVHCRF::initModel(DataSet &X)
{  
	pModel->setNumberOfSequenceLabels(X.searchNumberOfSequenceLabels());
	pFeatureGenerator->initFeatures(X, *pModel);
}


double ToolboxMVHCRF::test(DataSet& X, char* filenameOutput, char* filenameStats)
{ 
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
	int j = 0;
	int nbSeqLabels = pModel->getNumberOfSequenceLabels();
	iVector seqTruePos(nbSeqLabels);
	iVector seqTotalPos(nbSeqLabels);
	iVector seqTotalPosDetected(nbSeqLabels);

	int cnt=0;
	for(it = X.begin(); it != X.end(); it++) 
	{
		// Compute detected label
		dMatrix* matProbabilities = new dMatrix;
		int labelDetected = pEvaluator->computeSequenceLabel(
				*it,pModel,matProbabilities,getModel()->isMaxMargin());
		(*it)->setEstimatedProbabilitiesPerStates(matProbabilities);
		(*it)->setEstimatedSequenceLabel(labelDetected);

		// Read ground truth label
		int label = (*it)->getSequenceLabel();

		// optionally writes results in file
		if( fileOutput)
			(*fileOutput) << labelDetected << std::endl;

		// Update total of positive detections
		seqTotalPos[label]++;
		seqTotalPosDetected[labelDetected]++;
		if( label == labelDetected)
			seqTruePos[label]++;
	}
	// Print results
	if(fileStats)
	{
		(*fileStats) << "\nRegL1 [" << pModel->getRegL1FeatureTypes() << "] : " << pModel->getRegL1Sigma() << std::endl;
		(*fileStats) << "RegL2 [" << pModel->getRegL2FeatureTypes() << "] : " << pModel->getRegL2Sigma() << std::endl;
		(*fileStats) << std::endl << "Calculations per sequence:" << std::endl;
		(*fileStats) << "Label\tTrue+\tMarked+\tDetect+\tPrec.\tRecall\tF1" << std::endl;
	}
	double prec,recall;
	int SumTruePos = 0, SumTotalPos = 0, SumTotalPosDetected = 0;
	for(int i=0 ; i<nbSeqLabels ; i++) 
	{
		SumTruePos += seqTruePos[i]; SumTotalPos += seqTotalPos[i]; SumTotalPosDetected += seqTotalPosDetected[i];
		prec=(seqTotalPos[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPos[i]))/1000;
		recall=(seqTotalPosDetected[i]==0)?0:((double)(seqTruePos[i]*100000/seqTotalPosDetected[i]))/1000;
		if(fileStats)
			(*fileStats) << i << ":\t" << seqTruePos[i] << "\t" << seqTotalPos[i] << "\t" << seqTotalPosDetected[i] << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
	}
	prec=(SumTotalPos==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPos);
	recall=(SumTotalPosDetected==0)?0:((double)SumTruePos)*100.0/((double)SumTotalPosDetected);
	if(fileStats)
	{
		(*fileStats) << "-----------------------------------------------------------------------" << std::endl;
		(*fileStats) << "Ov:\t" << SumTruePos << "\t" << SumTotalPos << "\t" << SumTotalPosDetected << "\t" << prec << "\t" << recall << "\t" << 2*prec*recall/(prec+recall) << std::endl;
	}
	returnedF1value = 2*prec*recall/(prec+recall);

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


