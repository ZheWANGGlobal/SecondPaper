//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Belief Propagation (sum-product and max-product)
//
// Main structure of BP is implemented in computeBeliefs()
//
// TODO: Replace InferenceEngineBP.cpp with this class
//
// Yale Song (yalesong@csail.mit.edu)
// January, 2012

#include "inferenceengine.h"
#ifdef _OPENMP
#include <omp.h>
#endif  
 
InferenceEngineGBP::InferenceEngineGBP(): InferenceEngine()
{}

InferenceEngineGBP::~InferenceEngineGBP()
{}

void InferenceEngineGBP::computeBeliefs(Beliefs &beliefs, FeatureGenerator *fGen, 
	DataSequence *X, Model *m, int bComputePartition, int seqLabel, 
	bool bUseStatePerNodes, bool bMaxProduct)
{  
	if( bMaxProduct )
		throw HcrfBadModel("InferenceEngineGBP::computeBeliefs() MAP not implemented.");

	// Variable definition  
	int xi, xj, nbNodes, seqLength;
	std::map<int,BPNode*> nodes; // tree graph
	std::map<int,BPNode*>::iterator itm;
	std::list<BPNode*>::iterator itl;
	BPNode* root;
	iMatrix adjMat;
	iVector nbStates;
	dVector*  psi_i  = 0; // singleton potentials
	dMatrix** psi_ij = 0; // pairwise potentials
	dVector** msg    = 0; // messages

	if( m->isMultiViewMode() )
		m->getAdjacencyMatrixMVT(adjMat, fGen, X, seqLabel);
	else {
		uMatrix uAdjMat;
		m->getAdjacencyMatrix(uAdjMat, X);
		adjMat.resize(uAdjMat.getWidth(),uAdjMat.getHeight());
		for(xi=0; xi<uAdjMat.getHeight(); xi++)
			for(xj=0; xj<uAdjMat.getWidth(); xj++)
				adjMat(xi,xj) = uAdjMat(xi,xj);
	}

	nbNodes = adjMat.getHeight(); 
	seqLength = X->length();

	// Create a vector that contains nbStates
	nbStates.create(nbNodes);
	for(xi=0; xi<nbNodes; xi++)
		nbStates[xi] = (m->isMultiViewMode()) 
			? m->getNumberOfStatesMV(xi/seqLength) : m->getNumberOfStates();

	// Create BPGraph from adjMat
	for(xi=0; xi<nbNodes; xi++) {		
		BPNode* v = new BPNode(xi, nbStates[xi]);
		nodes.insert( std::pair<int,BPNode*>(xi,v) );
	}
	for(xi=0; xi<nbNodes; xi++) {
		for(xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			nodes[xi]->addNeighbor(nodes[xj]);
			nodes[xj]->addNeighbor(nodes[xi]);
		}
	}

	// Initialize  
	initMessages(msg, X, m, adjMat, nbStates);
	initBeliefs(beliefs, X, m, adjMat, nbStates);
	initPotentials(psi_i, psi_ij, fGen, X, m, adjMat, nbStates, seqLabel);
	
	// Message update
	root = nodes[0]; // any node can be the root node
	{
		for(itl=root->neighbors.begin(); itl!=root->neighbors.end(); itl++)
			collect(root, *itl, psi_i, psi_ij, msg);
		for(itl=root->neighbors.begin(); itl!=root->neighbors.end(); itl++)
			distribute(root, *itl, psi_i, psi_ij, msg);
	}
	updateBeliefs(beliefs, psi_i, psi_ij, msg, X, m, adjMat);

	// Clean up
	for(xi=0; xi<nbNodes; xi++) { 		
		delete[] msg[xi]; msg[xi] = 0; 
		delete[] psi_ij[xi]; psi_ij[xi] = 0;
	}
	delete[] msg; msg=0;
	delete[] psi_i; psi_i = 0;
	delete[] psi_ij;  psi_ij  = 0; 

	for(itm=nodes.begin(); itm!=nodes.end(); itm++) 
		delete (*itm).second; 
	nodes.clear();   
}

double InferenceEngineGBP::computePartition(FeatureGenerator *fGen, DataSequence *X, 
	Model *m, int seqLabel, bool bUseStatePerNodes, bool bMaxProduct)
{
	Beliefs beliefs;
	computeBeliefs(beliefs, fGen, X, m, false, seqLabel, bUseStatePerNodes); 

	return beliefs.partition;
}


int InferenceEngineGBP::computeSingleBelief(FeatureGenerator* fGen, Model* model, 
	DataSequenceRealtime* dataSequence, dVector* prob)
{ 
	throw HcrfNotImplemented("InferenceEngineGBP::computeSingleBelief() not implemented.\n"); 
}
  

/////////////////////////////////////////////////////////////////////////////////
// Private
void InferenceEngineGBP::collect(BPNode* xi, BPNode* xj, dVector* psi_i, dMatrix** psi_ij, dVector** m)
{
	std::list<BPNode*>::iterator it;
	for(it=xj->neighbors.begin(); it!=xj->neighbors.end(); it++) {
		if( xi->equal(*it) ) continue;
		collect(xj,*it,psi_i,psi_ij,m);
	}
	sendMessage(xj,xi,psi_i,psi_ij,m);		
}

void InferenceEngineGBP::distribute(BPNode* xi, BPNode* xj, dVector* psi_i, dMatrix** psi_ij, dVector** m)
{
	std::list<BPNode*>::iterator it;
	sendMessage(xi,xj,psi_i,psi_ij,m);
	for(it=xj->neighbors.begin(); it!=xj->neighbors.end(); it++) {
		if( xi->equal(*it) ) continue;
		distribute(xj,*it,psi_i,psi_ij,m);
	}
}

// m_ij(x_j) = sum_xi {psi(i)*psi(i,j)*prod_{u \in N(i)\j} {m_uj(xi)}}
void InferenceEngineGBP::sendMessage(BPNode* xi, BPNode* xj, dVector* psi_i, dMatrix** psi_ij, dVector** msg)
{   
	// potential(i)
	dVector Vi(psi_i[xi->id]);
	
	// potential(i,j)
	dMatrix Mij;
	if( xi->id < xj->id )
		Mij.set( psi_ij[xi->id][xj->id] );
	else {
		Mij.set( psi_ij[xj->id][xi->id] );
		Mij.transpose();
	}

	// prod_{u \in N(i)\j} {m_ui(xi)}	
	std::list<BPNode*>::iterator it;
	for(it=xi->neighbors.begin(); it!=xi->neighbors.end(); it++) {
		if( xj->equal(*it) ) continue;
		Vi.add( msg[(*it)->id][xi->id] );
	}
	
	logMultiply( Vi, Mij, msg[xi->id][xj->id] ); 
}  

void InferenceEngineGBP::initMessages(dVector**& msg, DataSequence* X, Model* m, iMatrix adjMat, iVector nbStates)
{
	int xi, xj, nbNodes, seqLength;
	nbNodes   = adjMat.getHeight();
	seqLength = X->length();

	msg = new dVector*[nbNodes];
	for(xi=0; xi<nbNodes; xi++) {
		msg[xi] = new dVector[nbNodes];
		for(xj=0; xj<nbNodes; xj++) 
			if( adjMat(xi,xj) ) {
				msg[xi][xj].create(nbStates[xj]);
			}
	}
}

void InferenceEngineGBP::initBeliefs(Beliefs& b, DataSequence* X, Model* m, iMatrix adjMat, iVector nbStates)
{
	int xi, xj, nbNodes, nbEdges, seqLength;
	nbNodes   = adjMat.getHeight();
	nbEdges   = nbNodes - 1;
	seqLength = X->length();

	b.belStates.resize(nbNodes);
	for(xi=0; xi<nbNodes; xi++) { 
		b.belStates[xi].create(nbStates[xi]);
	}

	b.belEdges.resize(nbEdges);	
	for(xi=0; xi<nbNodes; xi++) {
		for(xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			b.belEdges[adjMat(xi,xj)-1].create(nbStates[xj],nbStates[xi]);			
		}
	}
}

void InferenceEngineGBP::initPotentials(dVector*& psi_i, dMatrix**& psi_ij, 
	FeatureGenerator* fGen, DataSequence* X, Model* m, iMatrix adjMat, iVector nbStates, int seqLabel)
{
	int k, xi, xj, nbNodes, seqLength;
	nbNodes   = adjMat.getHeight();
	seqLength = X->length();

	// init singleton potentials
	psi_i = new dVector[nbNodes];
	for(xi=0; xi<nbNodes; xi++) {
		psi_i[xi].create(nbStates[xi]);
	}

	// init pairwise potentials
	psi_ij = new dMatrix*[nbNodes];
	for(xi=0; xi<nbNodes; xi++) {
		psi_ij[xi] = new dMatrix[nbNodes];
		for(xj=xi+1; xj<nbNodes; xj++) // ALWAYS (xj>xi)
			if( adjMat(xi,xj) )
				psi_ij[xi][xj].create(nbStates[xj],nbStates[xi]);
	} 
 
	//
	// Assign evidence to potentials
	int ThreadID = 0;
#if defined(_OPENMP)
	ThreadID = omp_get_thread_num();
	if( ThreadID >= nbThreadsMP ) ThreadID = 0;
#endif

	bool useVecFeaturesMP = true;
#if !defined(_VEC_FEATURES) && !defined(_OPENMP)
	useVecFeaturesMP = false;
#endif
	featureVector vecFeatures;
	featureVector myVecFeat = (useVecFeaturesMP) ? vecFeaturesMP[ThreadID] : vecFeatures;

	feature *f;
	dVector *lambda = m->getWeights(seqLabel); 	
	double val;

	// singleton potentials
	for(xi=0; xi<nbNodes; xi++) {
		fGen->getFeatures(myVecFeat,X,m,xi,-1,seqLabel);
		f = myVecFeat.getPtr();
		for(k=0; k<myVecFeat.size(); k++, f++) {
			val = (*lambda)[f->id] * f->value;
			psi_i[xi].addValue(f->nodeState,val);
		}
	}	
	// pairwise potentials
	for(xi=0; xi<nbNodes; xi++) {
		for(xj=xi+1; xj<nbNodes; xj++) {
			if( !adjMat(xi,xj) ) continue;
			fGen->getFeatures(myVecFeat,X,m,xj,xi,seqLabel);
			f = myVecFeat.getPtr();
			for(k=0; k<myVecFeat.size(); k++, f++) {
				val = (*lambda)[f->id] * f->value;
				psi_ij[xi][xj].addValue(f->prevNodeState,f->nodeState,val);
			}
		}
	}

}
 
// b_i(xi) = potential(i) * prod_{u \in N(i)}{m_ui}
// b_ij(xi,xj) = potential(i) * potential(j) * potential(i,j)
//			     * prod_{u \in N(i)\j}{m_ui} * prod_{u \in N(j)\i}{m_uj}
void InferenceEngineGBP::updateBeliefs(Beliefs& b, dVector* psi_i, dMatrix** psi_ij, 
	dVector** msg, DataSequence* X, Model* m, iMatrix adjMat)
{
	int xi, xj, xu, nbNodes, seqLength;
	dVector Vi, Vj;
	dMatrix Mij;
	
	nbNodes = adjMat.getHeight();
	seqLength = X->length();

	// b_i(xi) = potential(i) * prod_{u \in N(i)}{m_ui}
	for(xi=0; xi<nbNodes; xi++) {
		b.belStates[xi].set( psi_i[xi] ); 
		for(int xu=0; xu<nbNodes; xu++) {
			if( !adjMat(xu,xi) ) continue;
			b.belStates[xi].add( msg[xu][xi] );
		}	  
	}

	// b_ij(xi,xj) = potential(i) * potential(j) * potential(i,j)
	//			   * prod_{u \in N(i)\j}{m_ui} * prod_{u \in N(j)\i}{m_uj}	 
	for(xi=0; xi<nbNodes; xi++) {
		for(xj=xi+1; xj<nbNodes; xj++) { // xj starts from xi+1 because b_ij==b_ji
			if( !adjMat(xi,xj) ) continue;

			// potential(i) * prod_{u \in N(i)\j){m_ui}
			Vi.set( psi_i[xi] );
			for(xu=0; xu<nbNodes; xu++ ) {
				if( !adjMat(xu,xi) || xu==xj ) continue;
				Vi.add( msg[xu][xi] );
			}

			// potential(j) * prod_{u \in N(j)\i){m_uj}
			Vj.set( psi_i[xj] );
			for(xu=0; xu<nbNodes; xu++ ) {
				if( !adjMat(xu,xj) || xu==xi ) continue;
				Vj.add( msg[xu][xj] );
			}

			// (Vi*Vj*Mij) * potential(i,j)
			Mij.create(Vj.getLength(), Vi.getLength());
			logMultiply(Vi, Vj, Mij);
			Mij.add( psi_ij[xi][xj] );

			if( m->isMultiViewMode() )
				b.belEdges[adjMat(xi,xj)-1].set(Mij);
			else
				b.belEdges[xi].set(Mij);
		}
	}   

	// Normalize beliefs and compute partition
	unsigned int i;
	double logZ = b.belStates[0].logSumExp();
	b.partition = logZ;
	for(i=0; i<b.belStates.size(); i++) {
		b.belStates[i].add(-logZ);
		b.belStates[i].eltExp();
	}
	for(i=0; i<b.belEdges.size(); i++) {
		b.belEdges[i].add(-logZ);
		b.belEdges[i].eltExp();
	} 
}

int InferenceEngineGBP::logMultiplyMaxProd(dVector src_Vi, dMatrix src_Mij, dVector& dst_Vj)
{ 
	int max_idx=-1; double max_val=-DBL_MAX;
	dVector tmp_Vi;
	for(int c=0; c<src_Mij.getWidth(); c++) {
		max_idx = -1; max_val=-DBL_MAX; 
		tmp_Vi.set(src_Vi); 
		for(int r=0; r<src_Mij.getHeight(); r++)
			tmp_Vi[r] += src_Mij(r,c);
		for(int r=0; r<src_Mij.getHeight(); r++)
			if( tmp_Vi[r]>max_val ){ max_val=tmp_Vi[r]; max_idx=r; }
		dst_Vj[c] = max_val;
	}
	return max_idx;
} 

void InferenceEngineGBP::printBeliefs(Beliefs beliefs)
{
	for(int i=0; i<(int)beliefs.belStates.size(); i++) {
		for(int r=0; r<beliefs.belStates[i].getLength(); r++ )
			printf("%f\n", beliefs.belStates[i][r]);
		printf("------------------------------------------------------------\n");
	}

	for(int i=0; i<(int)beliefs.belEdges.size(); i++) {
		for(int r=0; r<beliefs.belEdges[i].getHeight(); r++) {
			for(int c=0; c<beliefs.belEdges[i].getWidth(); c++)
				printf("%f\t", beliefs.belEdges[i](r,c));
			printf("\n");
		}		
		printf("------------------------------------------------------------\n");
	}
}
  
///////////////////////////////////////////////////////////////////////////
// Implementation of the class BPNode
//  
BPNode::BPNode(int node_id, int node_size): id(node_id), size(node_size) {}
BPNode::~BPNode() {}
void BPNode::addNeighbor(BPNode* v) { neighbors.push_back(v); }
bool BPNode::equal(BPNode* v) { return id==v->id; }
void BPNode::print()
{
	std::list<BPNode*>::iterator it;
	printf("[%d] {", id); 
	for(it=neighbors.begin(); it!=neighbors.end(); it++) printf("%d ", (*it)->id); printf("}\n");
}