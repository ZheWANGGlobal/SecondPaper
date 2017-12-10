//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Belief Propagation with only two nodes
// 
// Yale Song (yalesong@csail.mit.edu)
// January, 2012

#include "inferenceengine.h" 

void InferenceEnginePiecewise::computeBeliefs(Beliefs& b, FeatureGenerator* fGen, DataSequence* X,
		Model* m, int y, int xi, int xj, iVector nbStates, bool bComputeMaxMargin)
{
	int k;	
	double val;

	feature *f;
	featureVector myVecFeat;
	dVector *w = m->getEdgeWeights(y);
	
	// Initialize beliefs
	b.belStates.resize(2);
	b.belEdges.resize(1);
	b.belStates[0].create(nbStates[xi]);
	b.belStates[1].create(nbStates[xj]);
	b.belEdges[0].create(nbStates[xj],nbStates[xi]);

	dVector m_ij(nbStates[xj]), m_ji(nbStates[xi]);
	dVector phi_i(nbStates[xi]), phi_j(nbStates[xj]);
	dMatrix phi_ij(nbStates[xj],nbStates[xi]), phi_ji(nbStates[xi],nbStates[xj]);

	// Compute potentials phi_i, phi_j, phi_ij
	fGen->getFeatures(myVecFeat,X,m,xi,-1,y); 
	f = myVecFeat.getPtr();
	for(k=0; k<myVecFeat.size(); k++, f++) {
		val = (*w)[f->id] * f->value;
		phi_i[f->nodeState] += val;
	} 
	fGen->getFeatures(myVecFeat,X,m,xj,-1,y);
	f = myVecFeat.getPtr();
	for(k=0; k<myVecFeat.size(); k++, f++) {
		val = (*w)[f->id] * f->value; 
		phi_j[f->nodeState] += val;
	} 
	fGen->getFeatures(myVecFeat,X,m,xj,xi,y);
	f = myVecFeat.getPtr();
	for(k=0; k<myVecFeat.size(); k++, f++) {
		val = (*w)[f->id] * f->value;
		phi_ij(f->prevNodeState,f->nodeState) += val;
		phi_ji(f->nodeState,f->prevNodeState) += val;
	}

	// Message update
	logMultiply(phi_i,phi_ij,m_ij);
	logMultiply(phi_j,phi_ji,m_ji);

	// Compute beliefs
	b.belStates[0].set(phi_i); b.belStates[0].add(m_ji);
	b.belStates[1].set(phi_j); b.belStates[1].add(m_ij);
	logMultiply(phi_i,phi_j,b.belEdges[0]); b.belEdges[0].add(phi_ij);
	
	// Normalize beliefs and compute partition
	b.partition = b.belStates[0].logSumExp();
	b.belStates[0].add(-b.partition); b.belStates[0].eltExp();
	b.belStates[1].add(-b.partition); b.belStates[1].eltExp();
	b.belEdges[0].add(-b.partition);  b.belEdges[0].eltExp();	 
}

