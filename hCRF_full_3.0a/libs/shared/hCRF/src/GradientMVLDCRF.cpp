//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of 
// GradientMVLDCRF Component
//
// Yale Song (yalesong@csail.mit.edu)
// October, 2011

#include "gradient.h"  

GradientMVLDCRF::GradientMVLDCRF(InferenceEngine* infEngine, FeatureGenerator* featureGen):
Gradient(infEngine, featureGen)
{}

double GradientMVLDCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin)
{ 
	if( bComputeMaxMargin )
		return computeGradientMaxMargin(vecGradient,m,X);
	else
		return computeGradientMLE(vecGradient,m,X);
}
 
double GradientMVLDCRF::computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X)
{
	////////////////////////////////////////////////////////////////////////////////////
	// Step 1 : Run Inference in each network to compute marginals
	Beliefs bel, belMasked;
	pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, true, -1, false);
	pInfEngine->computeBeliefs(belMasked, pFeatureGen, X, m, true, -1, true);

	// This is the value to be returned
	double f_value =  bel.partition - belMasked.partition;

	
	////////////////////////////////////////////////////////////////////////////////////
	// Step 2 : Update the gradient
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	if( vecGradient.getLength() != nbFeatures )
		vecGradient.create(nbFeatures); 
	
	int ThreadID = 0;
#if defined(_OPENMP)
	ThreadID = omp_get_thread_num();
	if( ThreadID >= nbThreadsMP ) ThreadID = 0;
#endif
		
	bool useVecFeaturesMP = false;
#if defined(_VEC_FEATURES) || defined(_OPENMP)
	useVecFeaturesMP = true;
#endif
	featureVector vecFeatures;
	featureVector myVecFeat = (useVecFeaturesMP) ? vecFeaturesMP[ThreadID] : vecFeatures;
	iMatrix adjMat;
	m->getAdjacencyMatrixMV(adjMat, X);
	
	int V = m->getNumberOfViews();
	int T = X->length(); 
	int nbNodes= V*T;

	feature* f; 
	
	// For xxLDCRF
	int seqLabel = -1;

	// Loop over nodes to compute features and update the gradient
	for(int xi=0; xi<nbNodes; xi++) {
		pFeatureGen->getFeatures(myVecFeat,X,m,xi,-1,seqLabel);			
		f = myVecFeat.getPtr();						
		for(int k=0; k<myVecFeat.size(); k++, f++) {  				
			// p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
			double gain = bel.belStates[xi][f->nodeState] - belMasked.belStates[xi][f->nodeState];
			vecGradient[f->id] -= gain * f->value;
		} 
	} 

	// Loop over edges to compute features and update the gradient
	for(int xi=0; xi<nbNodes; xi++) {
		for(int xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			pFeatureGen->getFeatures(myVecFeat,X,m,xj,xi,seqLabel);
			f = myVecFeat.getPtr();				
			for(int k=0; k<myVecFeat.size(); k++, f++) {
				// p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
				double gain = bel.belEdges[adjMat(xi,xj)-1](f->prevNodeState,f->nodeState)
							- belMasked.belEdges[adjMat(xi,xj)-1](f->prevNodeState,f->nodeState);
				vecGradient[f->id] -= gain * f->value;
			} 
		} 
	} 	
 
	//Return -log instead of log() [Moved to Gradient::ComputeGradient by LP]
	//vecGradient.negate(); 
	return f_value;
}



double GradientMVLDCRF::computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X)
{
	////////////////////////////////////////////////////////////////////////////////////
	// Step 1 : Run Inference in each network 
	Beliefs bel, belMasked;
	pInfEngine->computeBeliefs(bel, pFeatureGen, X, m, true, -1, false, true);
	pInfEngine->computeBeliefs(belMasked, pFeatureGen, X, m, true, -1, true, true);

	// This is the value to be returned
	double f_value = bel.partition - belMasked.partition;

	
	////////////////////////////////////////////////////////////////////////////////////
	// Step 2 : Update the gradient
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	if( vecGradient.getLength() != nbFeatures )
		vecGradient.create(nbFeatures); 
	
	int ThreadID = 0;
#if defined(_OPENMP)
	ThreadID = omp_get_thread_num();
	if( ThreadID >= nbThreadsMP ) ThreadID = 0;
#endif
		
	bool useVecFeaturesMP = false;
#if defined(_VEC_FEATURES) || defined(_OPENMP)
	useVecFeaturesMP = true;
#endif
	featureVector vecFeatures;
	featureVector myVecFeat = (useVecFeaturesMP) ? vecFeaturesMP[ThreadID] : vecFeatures;
	iMatrix adjMat;
	m->getAdjacencyMatrixMV(adjMat, X);
	
	int V = m->getNumberOfViews();
	int T = X->length(); 
	int nbNodes= V*T;

	feature* f; 
	
	// For xxLDCRF
	int seqLabel = -1;

	// Loop over nodes to compute features and update the gradient
	for(int xi=0; xi<nbNodes; xi++) {
		pFeatureGen->getFeatures(myVecFeat,X,m,xi,-1,seqLabel);			
		f = myVecFeat.getPtr();						
		for(int k=0; k<myVecFeat.size(); k++, f++) {  				
			// p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
			double gain = bel.belStates[xi][f->nodeState] - belMasked.belStates[xi][f->nodeState];
			vecGradient[f->id] -= gain * f->value;
		} 
	} 

	// Loop over edges to compute features and update the gradient
	for(int xi=0; xi<nbNodes; xi++) {
		for(int xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			pFeatureGen->getFeatures(myVecFeat,X,m,xj,xi,seqLabel);
			f = myVecFeat.getPtr();				
			for(int k=0; k<myVecFeat.size(); k++, f++) {
				// p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
				double gain = bel.belEdges[adjMat(xi,xj)-1](f->prevNodeState,f->nodeState)
							- belMasked.belEdges[adjMat(xi,xj)-1](f->prevNodeState,f->nodeState);
				vecGradient[f->id] -= gain * f->value;
			} 
		} 
	} 	
  
	return f_value;
}

