//-------------------------------------------------------------
// Hidden Conditional Random Field Library - GradientHCRF
// Component
//
//	February 2, 2006

#include "gradient.h"

GradientHCRF::GradientHCRF(InferenceEngine* infEngine, 
						   FeatureGenerator* featureGen) 
  : Gradient(infEngine, featureGen)
{
}


double GradientHCRF::computeGradient(dVector& vecGradient, Model* m, DataSequence* X, bool bComputeMaxMargin)
{
	if( bComputeMaxMargin ) 
		return computeGradientMaxMargin(vecGradient,m,X);
	else
		return computeGradientMLE(vecGradient,m,X);
}

double GradientHCRF::computeGradientMLE(dVector& vecGradient, Model* m, DataSequence* X)
{
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	int NumSeqLabels=m->getNumberOfSequenceLabels();
	//Get adjency matrix
	uMatrix adjMat;
	m->getAdjacencyMatrix(adjMat, X);
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);
	dVector Partition;
	Partition.resize(1,NumSeqLabels);
	std::vector<Beliefs> ConditionalBeliefs(NumSeqLabels);

	// Step 1 : Run Inference in each network to compute marginals conditioned on Y
	for(int i=0;i<NumSeqLabels;i++)
	{
		pInfEngine->computeBeliefs(ConditionalBeliefs[i],pFeatureGen, X, m, true,i);
		Partition[i] = ConditionalBeliefs[i].partition;
	}
	double f_value = Partition.logSumExp() - Partition[X->getSequenceLabel()];
	// Step 2: Compute expected values for feature nodes conditioned on Y
#if !defined(_VEC_FEATURES) && !defined(_OPENMP)
	featureVector* vecFeatures;
#endif
#if defined(_OPENMP)
	int ThreadID = omp_get_thread_num();
	if (ThreadID >= nbThreadsMP)
		ThreadID = 0;
#else
	int ThreadID = 0;
#endif
	double value;
	dMatrix CEValues;
	CEValues.resize(nbFeatures,NumSeqLabels);
	//Loop over nodes to compute features and update the gradient
	for(int j=0;j<NumSeqLabels;j++) {//For every labels
		for(int i = 0; i < X->length(); i++) {//For every nodes
#if defined(_VEC_FEATURES) || defined(_OPENMP)
			pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,i,-1,j);
			// Loop over features
			feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
			for(int k = 0; k < vecFeaturesMP[ThreadID].size(); k++, pFeature++)
#else
		vecFeatures =pFeatureGen->getFeatures(X,m,i,-1,j);
		  // Loop over features
		  feature* pFeature = vecFeatures->getPtr();
		  for(int k = 0; k < vecFeatures->size(); k++, pFeature++)
#endif
			{   
                //p(s_i=s|x,Y) * f_k(i,s,x,y) 
				value=ConditionalBeliefs[j].belStates[i][pFeature->nodeState] * pFeature->value;
				CEValues.setValue(j,pFeature->globalId, CEValues(j,pFeature->globalId) + value); // one row for each Y
			}// end for every feature
		}// end for every node
	}// end for ever Sequence Label
	// Step 3: Compute expected values for edge features conditioned on Y
	//Loop over edges to compute features and update the gradient
	for(int j=0;j<NumSeqLabels;j++){
		int edgeIndex = 0;
	    for(int row = 0; row < X->length(); row++){
			// Loop over all rows (the previous node index)
		    for(int col = row; col < X->length() ; col++){
				//Loop over all columns (the current node index)
				if(adjMat(row,col) == 1) {
					//Get nodes features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
					pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X,m,col,row,j);
					// Loop over features
					feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
					for(int k = 0; k < vecFeaturesMP[ThreadID].size(); k++, pFeature++)
#else
					vecFeatures = pFeatureGen->getFeatures(X,m,col,row,j);
					// Loop over features
					feature* pFeature = vecFeatures->getPtr();
					for(int k = 0; k < vecFeatures->size(); k++, pFeature++)
#endif
					{
                        //p(y_i=s1,y_j=s2|x,Y)*f_k(i,j,s1,s2,x,y) 
						value=ConditionalBeliefs[j].belEdges[edgeIndex](pFeature->prevNodeState,pFeature->nodeState) * pFeature->value;
						CEValues.setValue(j,pFeature->globalId, CEValues(j,pFeature->globalId) + value);
					}
					edgeIndex++;
				}
			}
		}
	}
	// Step 4: Compute Joint Expected Values
	dVector JointEValues;
	JointEValues.resize(1,nbFeatures);
	JointEValues.set(0);
	dVector rowJ;
	rowJ.resize(1,nbFeatures);
	dVector GradientVector;
	double sumZLog=Partition.logSumExp();
	for (int j=0;j<NumSeqLabels;j++)
	 {
         CEValues.getRow(j, rowJ);
		 rowJ.multiply(exp(Partition.getValue(j)-sumZLog));
		 JointEValues.add(rowJ);
	 }
  // Step 5 Compute Gradient as Exi[i,*,*] -Exi[*,*,*], that is difference
  // between expected values conditioned on Sequence Labels and Joint expected
  // values
	 CEValues.getRow(X->getSequenceLabel(), rowJ); // rowJ=Expected value
												   // conditioned on Sequence
												   // label Y
	// [Negation moved to Gradient::ComputeGradient by LP]
//	 rowJ.negate(); 
	 JointEValues.negate();
	 rowJ.add(JointEValues);
	 vecGradient.add(rowJ);
	 return f_value;
}


double GradientHCRF::computeGradientMaxMargin(dVector& vecGradient, Model* m, DataSequence* X)
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
	if( f_value <= 0 ) 	return 0;


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
	uMatrix adjMat;
	m->getAdjacencyMatrix(adjMat, X);
	
	int nbNodes= X->length(); 
	int xi, xj, k;
	feature* f;	 

	// Viterbi Decoding 
	uVector viterbi_max(nbNodes), viterbi_truth(nbNodes); 
	
	y = max_y;	
	for(xi=0; xi<nbNodes; xi++) {
		max_h=-1; max_val=-DBL_MAX;
		for(h=0; h<condBeliefs[y].belStates[xi].getLength(); h++) {
			if( condBeliefs[y].belStates[xi][h] > max_val ) {
				max_h = h; 
				max_val = condBeliefs[y].belStates[xi][h];
		}	}
		viterbi_max[xi] = max_h; 
	}

	y = X->getSequenceLabel();	
	for(xi=0; xi<nbNodes; xi++) {
		max_h=-1; max_val=-DBL_MAX;
		for(h=0; h<condBeliefs[y].belStates[xi].getLength(); h++) {
			if( condBeliefs[y].belStates[xi][h] > max_val ) {
				max_h = h; 
				max_val = condBeliefs[y].belStates[xi][h];
		}	}
		viterbi_truth[xi] = max_h; 
	}

	// G
	dVector grad_max(nbFeatures), grad_truth(nbFeatures); 
	for(xi=0, xj=1; xi<nbNodes; xi++, xj++) {				
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
		if( xj<nbNodes ) {
			y = max_y;
			pFeatureGen->getFeatures(myVecFeat,X,m,xj,xi,y);
			for(k=0, f=myVecFeat.getPtr(); k<myVecFeat.size(); k++, f++)
				if( f->prevNodeState==viterbi_max[xi] && f->nodeState==viterbi_max[xj] )
					grad_max[f->globalId] 
						+= condBeliefs[y].belEdges[xi](f->prevNodeState,f->nodeState) * f->value;
			
			y = X->getSequenceLabel();
			pFeatureGen->getFeatures(myVecFeat,X,m,xj,xi,y);
			for(k=0, f=myVecFeat.getPtr(); k<myVecFeat.size(); k++, f++)
				if( f->prevNodeState==viterbi_truth[xi] && f->nodeState==viterbi_truth[xj] )
					grad_truth[f->globalId]  
						+= condBeliefs[y].belEdges[xi](f->prevNodeState,f->nodeState) * f->value;
		}
	} 	 
	 
	vecGradient.add(grad_truth);
	vecGradient.subtract(grad_max);

	return f_value; 
}

