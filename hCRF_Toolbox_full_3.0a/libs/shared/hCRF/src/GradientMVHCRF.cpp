//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of 
// GradientMVHCRF Component
//
// Yale Song (yalesong@csail.mit.edu)
// July, 2011

#include "gradient.h" 

GradientMVHCRF::GradientMVHCRF(InferenceEngine* ie, FeatureGenerator* fg): Gradient(ie, fg)
{
}

///////////////////////////////////////////////////////////////////////////
// PUBLIC
//
double GradientMVHCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin) 
{
	if( bComputeMaxMargin )
		return computeGradientMaxMargin(vecGradient,m,X);
	else
		return computeGradientMLE(vecGradient,m,X);
}



///////////////////////////////////////////////////////////////////////////
// PRIVATE
//
double GradientMVHCRF::computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X)
{    
	double f_value=0; // return value

	////////////////////////////////////////////////////////////////////////////////////
	// Step 1 : Run Inference in each network to compute marginals conditioned on Y
 	int nbSeqLabels = m->getNumberOfSequenceLabels();
	std::vector<Beliefs> condBeliefs(nbSeqLabels);	
	dVector Partition(nbSeqLabels);
	 
	for(int y=0; y<nbSeqLabels; y++) 
	{ 
		pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y);
		Partition[y] = condBeliefs[y].partition;; 
	} 
	
	////////////////////////////////////////////////////////////////////////////////////
	// Step 2 : Compute expected values for node/edge features conditioned on Y
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	dMatrix condEValues(nbFeatures, nbSeqLabels);
	
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
	double val;
	int y, k, xi, xj;
	
	for(y=0; y<nbSeqLabels; y++) 
	{ 
		// Loop over nodes to compute features and update the gradient
		for(xi=0; xi<nbNodes; xi++) {
			pFeatureGen->getFeatures(myVecFeat,X,m,xi,-1,y);			
			f = myVecFeat.getPtr();						
			for(k=0; k<myVecFeat.size(); k++, f++) {  				
				// p(h^v_t=a|x,y) * f_k(v,t,a,x,y)
				val = condBeliefs[y].belStates[xi][f->nodeState] * f->value;
				condEValues.addValue(y, f->globalId, val);
			} 
		} 

		// Loop over edges to compute features and update the gradient
		for(xi=0; xi<nbNodes; xi++) {
			for(xj=xi+1; xj<nbNodes; xj++) {
				if( !adjMat(xi,xj) ) continue;
				pFeatureGen->getFeatures(myVecFeat,X,m,xj,xi,y);
				f = myVecFeat.getPtr();				
				for(k=0; k<myVecFeat.size(); k++, f++) {
					// p(h^vi_ti=a,h^vj_tj=b|x,y) * f_k(vi,ti,vj,tj,x,y)
					val = condBeliefs[y].belEdges[adjMat(xi,xj)-1]
							(f->prevNodeState,f->nodeState) * f->value;
					condEValues.addValue(y, f->globalId, val);
				} 
			} 
		} 	
	} 

	////////////////////////////////////////////////////////////////////////////////////
	// Step 3: Compute Joint Expected Values
	dVector JointEValues(nbFeatures);
	dVector rowJ(nbFeatures);  // expected value conditioned on seqLabel Y
	double sumZLog = Partition.logSumExp();
	for (int y=0; y<nbSeqLabels; y++) 
	{
		condEValues.getRow(y, rowJ);
		rowJ.multiply( exp(Partition[y]-sumZLog) );
		JointEValues.add(rowJ);
	}
	
	////////////////////////////////////////////////////////////////////////////////////
	// Step 4 Compute Gradient as Exi[i,*,*] - Exi[*,*,*], that is the difference between 
	// expected values conditioned on seqLabel Y and joint expected values	
	if( vecGradient.getLength() != nbFeatures )
		vecGradient.create(nbFeatures);

	condEValues.getRow(X->getSequenceLabel(), rowJ); 
	// rowJ.negate(); // [Negation moved to Gradient::ComputeGradient by LP]
	JointEValues.negate();
	rowJ.add(JointEValues);
	vecGradient.add(rowJ);  

	// MLE: return log(sum_y' p(y'|xi)) - log(p(yi|xi)})	
	f_value = Partition.logSumExp() - Partition[X->getSequenceLabel()]; 
	return f_value;
}


double GradientMVHCRF::computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X)
{  
	int y, h, max_y, max_h, loss;
	double val, max_val, f_value;

	////////////////////////////////////////////////////////////////////////////////////
	// Step 1 : Run Inference in each network  
 	int nbSeqLabels = m->getNumberOfSequenceLabels();
	std::vector<Beliefs> condBeliefs(nbSeqLabels);	
	dVector Partition(nbSeqLabels); 

	for(y=0; y<nbSeqLabels; y++) {
		pInfEngine->computeBeliefs(condBeliefs[y], pFeatureGen, X, m, true, y, false, true);  					
		Partition[y] = condBeliefs[y].partition;
	}  

	// Find max_y
	max_y = -1; max_val = -DBL_MAX;
	for(y=0; y<nbSeqLabels; y++) { 
		loss = (y!=X->getSequenceLabel()); // 0-1 loss
		val = (double)loss + Partition[y];		
		if( val > max_val ) { max_y=y; max_val = val; }
	} 	

	// F
	f_value = max_val - Partition[X->getSequenceLabel()];
	if( f_value <= 0 ) return 0;

	
	////////////////////////////////////////////////////////////////////////////////////
	// Step 2 : Compute subgradient
	int nbFeatures = pFeatureGen->getNumberOfFeatures(); 
	
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
	int xi, xj, k;

	
	// Viterbi Decoding 
	uVector viterbi_max(nbNodes), viterbi_truth(nbNodes); 
	
	y = max_y;	
	for(xi=0; xi<nbNodes; xi++) {
		max_h=-1; max_val=-DBL_MAX;
		for(h=0; h<condBeliefs[y].belStates[xi].getLength(); h++) {
			if( condBeliefs[y].belStates[xi][h] > max_val ) {
				max_h   = h; 
				max_val = condBeliefs[y].belStates[xi][h];
		}	}
		viterbi_max[xi] = max_h; 
	}

	y = X->getSequenceLabel();	
	for(xi=0; xi<nbNodes; xi++) {
		max_h=-1; max_val=-DBL_MAX;
		for(h=0; h<condBeliefs[y].belStates[xi].getLength(); h++) {
			if( condBeliefs[y].belStates[xi][h] > max_val ) {
				max_h   = h; 
				max_val = condBeliefs[y].belStates[xi][h];
		}	}
		viterbi_truth[xi] = max_h; 
	}

	// G	 
	dVector grad_max(nbFeatures), grad_truth(nbFeatures); 
	for(xi=0; xi<nbNodes; xi++) {	
		// ---------- Singleton Features ---------- 
		y = max_y;
		pFeatureGen->getFeatures(myVecFeat,X,m,xi,-1,y);
		for(k=0, f=myVecFeat.getPtr(); k<myVecFeat.size(); k++, f++)
			if( f->nodeState == viterbi_max[xi] )  
				grad_max[f->globalId]   
					+= condBeliefs[y].belStates[xi][f->nodeState] * f->value;  
		
		y = X->getSequenceLabel();
		pFeatureGen->getFeatures(myVecFeat,X,m,xi,-1,y);
		for(k=0, f=myVecFeat.getPtr(); k<myVecFeat.size(); k++, f++)
			if( f->nodeState == viterbi_truth[xi] )
				grad_truth[f->globalId] 
					+= condBeliefs[y].belStates[xi][f->nodeState] * f->value;
	
		// ---------- Pairwise Features ---------- 
		for(xj=xi+1; xj<nbNodes; xj++) {
			if( adjMat(xi,xj)==0 ) continue;
			y = max_y;
			pFeatureGen->getFeatures(myVecFeat,X,m,xj,xi,y);
			for(k=0, f=myVecFeat.getPtr(); k<myVecFeat.size(); k++, f++) 
				if( f->prevNodeState==viterbi_max[xi] && f->nodeState==viterbi_max[xj] )
					grad_max[f->globalId] += condBeliefs[y].belEdges[adjMat(xi,xj)-1]
											(f->prevNodeState,f->nodeState) * f->value;
			
			y = X->getSequenceLabel();
			pFeatureGen->getFeatures(myVecFeat,X,m,xj,xi,y);
			for(k=0, f=myVecFeat.getPtr(); k<myVecFeat.size(); k++, f++) 
				if( f->prevNodeState==viterbi_truth[xi] && f->nodeState==viterbi_truth[xj] )
					grad_truth[f->globalId] += condBeliefs[y].belEdges[adjMat(xi,xj)-1]
											(f->prevNodeState,f->nodeState) * f->value;			
		} 
	} 	   
	vecGradient.add(grad_truth);
	vecGradient.subtract(grad_max);
	return f_value; 
 
 
}

 