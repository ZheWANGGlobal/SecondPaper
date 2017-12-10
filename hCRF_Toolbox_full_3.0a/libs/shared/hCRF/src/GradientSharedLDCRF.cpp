#include "gradient.h"
#include <vector>
#include <stdio.h>
#include <assert.h>

double GradientSharedLDCRF::computeGradient(dVector& vecGradient, Model* m,
										  DataSequence* X, bool bComputeMaxMargin)
{
	if( bComputeMaxMargin )
		throw HcrfNotImplemented("GradientLDCRF for max-margin is not implemented");

   //Check the size of vecGradient
   int nbFeatures = pFeatureGen->getNumberOfFeatures();
   if(vecGradient.getLength() != nbFeatures) {
	  vecGradient.create(nbFeatures);
   } 
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
   Beliefs bel_xy, bel_x;
   pInfEngine->computeBeliefs(bel_x, pFeatureGen, X, m, false, -1, false);
   pInfEngine->computeBeliefs(bel_xy, pFeatureGen, X, m, true, -1, true);
   int chainL = X->length();
   for(int i = 0; i < chainL; i++) {
#if defined(_VEC_FEATURES) || defined(_OPENMP)
	   //Get nodes features
	   pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X, m, i, -1);
	   // Loop over features (this loop over a and k at the same time for node
	   // feature. We need to compute (but we dont take into account the value
	   // (alwayse 1) We compute p(y_i = label \given X)
	   feature* endOfVector = vecFeaturesMP[ThreadID].getPtr() + vecFeaturesMP[ThreadID].size();
	   // We can now compute the coeff that come in front of the expression
	   for(feature* pFeature = vecFeaturesMP[ThreadID].getPtr(); 
		   pFeature < endOfVector ; pFeature++) {
#else
	  //Get nodes features
	  vecFeatures = pFeatureGen->getFeatures(X, m, i, -1);
	  // Loop over features (this loop over a and k at the same time for node
	  // feature. We need to compute (but we dont take into account the value
	  // (alwayse 1) We compute p(y_i = label \given X)
	  feature* endOfVector = vecFeatures->getPtr() + vecFeatures->size();
	  // We can now compute the coeff that come in front of the expression
	  for(feature* pFeature = vecFeatures->getPtr(); 
		  pFeature < endOfVector ; pFeature++) {
#endif
		   // This is a RawWindowFeatures
		   vecGradient[pFeature->id] += bel_xy.belStates[i][pFeature->nodeState] * pFeature->value;
		   vecGradient[pFeature->id] -= bel_x.belStates[i][pFeature->nodeState] * pFeature->value;
	   }
	   //Get Edge features features
	   if (i>0) {
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		  pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X, m, i, i-1);
		   // Loop over features
		   endOfVector = vecFeaturesMP[ThreadID].getPtr() + vecFeaturesMP[ThreadID].size();
		   for(feature* pFeature = vecFeaturesMP[ThreadID].getPtr(); pFeature < endOfVector ; pFeature++) {
#else
		  vecFeatures = pFeatureGen->getFeatures(X,m, i, i-1);
			// Loop over features
		  endOfVector = vecFeatures->getPtr() + vecFeatures->size();
		  for(feature* pFeature = vecFeatures->getPtr(); pFeature < endOfVector ; pFeature++) {
#endif
			   vecGradient[pFeature->id] += bel_xy.belEdges[i-1](pFeature->prevNodeState, pFeature->nodeState) * pFeature->value;
			   vecGradient[pFeature->id] -= bel_x.belEdges[i-1](pFeature->prevNodeState, pFeature->nodeState) * pFeature->value;
		   }
	   }
	   // Finally, we get the edges between the Y and the X
#if defined(_VEC_FEATURES) || defined(_OPENMP)
	   pFeatureGen->getFeatures(vecFeaturesMP[ThreadID], X, m, i, i+chainL);
	   // Loop over features
	   endOfVector = vecFeaturesMP[ThreadID].getPtr() + vecFeaturesMP[ThreadID].size();
	   for(feature* pFeature = vecFeaturesMP[ThreadID].getPtr(); pFeature < endOfVector ; pFeature++) 
#else
	  vecFeatures = pFeatureGen->getFeatures(X, m, i, i+chainL);
	  // Loop over features
	  endOfVector = vecFeatures->getPtr() + vecFeatures->size();
	  for(feature* pFeature = vecFeatures->getPtr(); pFeature < endOfVector ; pFeature++)
#endif
	  {
		   vecGradient[pFeature->id] += bel_xy.belEdges[i+chainL](pFeature->prevNodeState, pFeature->nodeState) * pFeature->value;
		   vecGradient[pFeature->id] -= bel_x.belEdges[i+chainL](pFeature->prevNodeState, pFeature->nodeState) * pFeature->value;
	   }
   }
   //[Moved to Gradient::ComputeGradient by LP]
  // vecGradient.negate();
   return bel_x.partition - bel_xy.partition;
}

