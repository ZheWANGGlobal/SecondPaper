#include "gradient.h"

GradientHMMPerceptron::GradientHMMPerceptron(InferenceEnginePerceptron* infEngine, 
											 FeatureGenerator* featureGen)
: GradientPerceptron(infEngine, featureGen)
{
}


double GradientHMMPerceptron::computeGradient(dVector& vecGradient, Model* m, 
									DataSequence* X, bool bComputeMaxMargin)
{
	if( bComputeMaxMargin )
		throw HcrfNotImplemented("GradientHMMPerceptron for max-margin is not implemented");

	double updated = 0;
	//compute beliefs
	iVector viterbiPath;
	pInfEngine->computeViterbiPath(viterbiPath,pFeatureGen, X, m, -1, false);
			
	//Get adjency matrix
	uMatrix adjMat;
	m->getAdjacencyMatrix(adjMat, X);
	//Check the size of vecGradient
	int nbFeatures = pFeatureGen->getNumberOfFeatures();
	if(vecGradient.getLength() != nbFeatures)
		vecGradient.create(nbFeatures);
#if defined(_VEC_FEATURES) || defined(_OPENMP)
	featureVector vecFeatures;
#else
	featureVector* vecFeatures;
#endif
	//Loop over nodes to compute features and update the gradient
	for(int i = 0; i < X->length(); i++)
	{
		// Read the label for this state
		int s = X->getStateLabels(i);
		int p = viterbiPath.getValue(i);
		if(s == p)
			continue;
		updated = 1;
		//Get nodes features
		// Loop over features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		pFeatureGen->getFeatures(vecFeatures, X,m,i,-1);
		feature* pFeature = vecFeatures.getPtr();
		for (int j = 0; j < vecFeatures.size() ; j++, pFeature++) {
#else
		vecFeatures = pFeatureGen->getFeatures(X,m,i,-1);
		feature* pFeature = vecFeatures->getPtr();
		for (int j = 0; j < vecFeatures->size() ; j++, pFeature++) {
#endif
			// If feature has same state label as the label from the
			// dataSequence, then add this to the gradient
			if(pFeature->nodeState == s)
				vecGradient[pFeature->id] += pFeature->value;
			if(pFeature->nodeState == p)
				vecGradient[pFeature->id] -= pFeature->value;			
		}
	}
	//Loop over edges to compute features and update the gradient	
	for(int row = 0; row < X->length(); row++) // Loop over all rows (the previous node index)
	{
		for(int col = row + 1; col < X->length() ; col++) //Loop over all columns (the current node index)
		{
			if(adjMat(row,col) == 1)
			{
				int s1 = X->getStateLabels(row);
				int s2 = X->getStateLabels(col);
				int p1 = viterbiPath.getValue(row);
				int p2 = viterbiPath.getValue(col);
				if(p1==s1 && p2==s2)
					continue;
				updated = 1;
				//Get edge features
#if defined(_VEC_FEATURES) || defined(_OPENMP)
				pFeatureGen->getFeatures(vecFeatures, X,m,col,row);
				feature* pFeature = vecFeatures.getPtr();
				for (int j = 0; j < vecFeatures.size() ; j++, pFeature++) {
#else
				vecFeatures = pFeatureGen->getFeatures(X,m,col,row);
				feature* pFeature = vecFeatures->getPtr();
				for (int j = 0; j < vecFeatures->size() ; j++, pFeature++) {
#endif
					// ++ Forward edge ++
					// If edge feature has same state labels as the labels from the dataSequence, then add it to the gradient
					if(pFeature->nodeState == s2 && pFeature->prevNodeState == s1)
						vecGradient[pFeature->id] += pFeature->value;
					if(pFeature->nodeState == p2 && pFeature->prevNodeState == p1)
						vecGradient[pFeature->id] += pFeature->value;
				}			
			}
		}
	}	
	return updated;
}
