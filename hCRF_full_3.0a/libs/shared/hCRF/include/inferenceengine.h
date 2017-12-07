//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Inference Engine
//
//	January 30, 2006

#ifndef INFERENCEENGINE_H
#define INFERENCEENGINE_H

//Standard Template Library includes
#include <vector>
#include <list>

#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <map>
#include <algorithm>

//hCRF Library includes
#include "hcrfExcep.h"

#include "featuregenerator.h"
#include "matrix.h"

//#define INF_VALUE -DBL_MAX
#define INF_VALUE 1e100

#if defined(__VISUALC__)||defined(__BORLAND__)
    #define wxFinite(n) _finite(n)
#elseif defined(__GNUC__)
    #define wxFinite(n) finite(n)
#else
    #define wxFinite(n) ((n) == (n))
#endif

struct Beliefs {
public:
    std::vector<dVector> belStates;
    std::vector<dMatrix> belEdges;
    double partition;
    Beliefs()
    :belStates(), belEdges(), partition(0.0) {};    
}; 

class InferenceEngine
{
  public:
    //Constructor/Destructor
    InferenceEngine();
    virtual ~InferenceEngine();

    virtual void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,
                               DataSequence* X, Model* m,
                               int bComputePartition,int seqLabel=-1,
                               bool bUseStatePerNodes = false, 
							   bool bMaxProduct=false)=0;
    virtual double computePartition(FeatureGenerator* fGen,DataSequence* X,
                               Model* m,int seqLabel=-1,
                               bool bUseStatePerNodes = false, 
							   bool bMaxProduct=false)=0;
	virtual int computeSingleBelief(FeatureGenerator* fGen, Model* model,
								DataSequenceRealtime* dataSequence, dVector* prob) = 0;
	virtual void setMaxNumberThreads(int maxThreads);
  protected:
    // Private function that are used as utility function for several
    // beliefs propagations algorithms.
    int CountEdges(const uMatrix& AdjacencyMatrix, int nbNodes) const;

    void computeLogMi(FeatureGenerator* fGen, Model* model, DataSequence* X,
                      int i, int seqLabel, dMatrix& Mi_YY, dVector& Ri_Y,
                      bool takeExp, bool bUseStatePerNodes) ;
    void LogMultiply(dMatrix& Potentials,dVector& Beli, dVector& LogAB);
	
	void logMultiply(dVector src_Vi, dMatrix src_Mij, dVector& dst_Vj);
	void logMultiply(dVector src_Vi, dVector src_Vj, dMatrix& dst_Mij);

	featureVector *vecFeaturesMP;
	int nbThreadsMP;

};


class InferenceEngineBP: public InferenceEngine {
  public:
    //Constructor/Destructor
    InferenceEngineBP();
    InferenceEngineBP(const InferenceEngineBP&);
    InferenceEngineBP& operator=(const InferenceEngineBP&);
    ~InferenceEngineBP();

    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen, DataSequence* X,
						Model* crf, int bComputePartition, int seqLabel=-1,
						bool bUseStatePerNodes = false,bool bMaxProduct=false);
    double computePartition(FeatureGenerator* fGen,DataSequence* X, Model* crf,
						int seqLabel=-1, bool bUseStatePerNodes = false, bool bMaxProduct=false);
	int computeSingleBelief(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob); 
  private:
    void MakeLocalEvidence(dMatrix& LogLocalEvidence, FeatureGenerator* fGen, 
                           DataSequence* X, Model* crf, int seqLabel);
    void MakeEdgeEvidence(std::vector<dMatrix>& LogEdgeEvidence, 
                          FeatureGenerator* fGen,DataSequence* X, Model* crf, 
                          int seqLabel);
    int GetEdgeMatrix(int nodeI, int nodeJ, dMatrix& EvidenceEdgeIJ, 
                      std::vector<dMatrix>& EdgePotentials);
    void TreeInfer(Beliefs& bel, FeatureGenerator* fGen, DataSequence* X,
                   iVector &PostOrder, iVector &PreOrder, Model* crf, 
                   int seqLabel, bool bUseStatePerNodes);
    int computeEdgeBel(Beliefs& bel, std::vector<dMatrix> Messages, 
                       std::vector<dMatrix>& LogEdgeEvidence);
    double computeZ(int i, int j, dVector& lastm, dMatrix& LogLocalEvidence);

    int children(int NodeNumber, iVector& Child, uMatrix& Ad);
    int parents(int NodeNumber,iVector& Parent,uMatrix& Ad);
    int neighboor(int NodeNumber,iVector& Neighboors,uMatrix& Ad);
    int dfs(iVector& Pred, iVector& Post);
    void Dfs_Visit(int u,uMatrix& Ad);
    int findNbrs(int nodeI, iVector& NBRS);
    void BuildEdgeMatrix2();

    void GetRow (dMatrix& M, int j, dVector& rowfVectorj);
    void PrintfVector(dVector& v1);
    double MaxValue(dVector& v1);
    void Transpose(dMatrix& M, dMatrix& MTranspose);
    void LogNormalise(dVector& OldV, dVector& NewV);
    void LogNormalise2(dMatrix& OldM, dMatrix& NewM);
    void repMat1(dVector& beli,int nr, dMatrix& Result);
    void repMat2(dVector& beli,int nr, dMatrix& Result);

    // Global Variables ( for the inference helper function that gets postorder
    int white_global, gray_global, black_global;
    int sizepre_global, sizepost_global;
    int time_stamp_global, cycle_global;
    iVector color_global, d_global, f_global;
    iVector pre_global, post_global, pred_global;
    int NNODES;
    int NEDGES;
    int NFEATURES;
    uMatrix AdjacencyMatrix;
    dVector *theta;
    iMatrix EMatrix;
    uMatrix Tree;

    int NSTATES;
};



class InferenceEngineDC:public InferenceEngine
{
/**
This class is used for beliefs propagation on a tree composed of chain
of hidden states where each is connected to labels:

y1   y2   y3   y4   y5   y6
|    |    |    |    |    |
h1---h2---h3---h4---h5---h6
|    |    |    |    |    |
X    X    X    X    X    X

X represents the observations. 
**/   
  public:
    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,DataSequence* X,
                        Model* crf, int bComputePartition,int seqLabel=-1,
                        bool bUseStatePerNodes = false,bool bMaxProduct=false);
    double computePartition(FeatureGenerator* fGen, DataSequence* X,
                        Model* crf, int seqLabel=-1,
                        bool bUseStatePerNodes = false,bool bMaxProduct=false);
	int computeSingleBelief(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob); 
  protected:
    void computeObsMsg(FeatureGenerator* fGen, Model* model, DataSequence* X,
                       int i, dMatrix& Mi_HY, dVector& Ri_Y, dVector& Pi_Y,
                       bool takeExp, bool bUseStatePerNodes);
    void computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen,
                           DataSequence* X, Model* model,
                           int bComputePartition,int seqLabel,
                           bool bUseStatePerNodes);
    void computeBeliefsLinear(Beliefs& bel, FeatureGenerator* fGen,
						  DataSequence* X, Model* model,
						  int bComputePartition,
						  int seqLabel, bool bUseStatePerNodes);
};

class InferenceEngineFB:public InferenceEngine
{
public:
    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,DataSequence* X,
                        Model* crf, int bComputePartition,int seqLabel=-1,
                        bool bUseStatePerNodes = false,bool bMaxProduct=false);
    double computePartition(FeatureGenerator* fGen, DataSequence* X,
                        Model* crf, int seqLabel=-1,
                        bool bUseStatePerNodes = false,bool bMaxProduct=false);	
	int computeSingleBelief(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob); 
private:
    void computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen,
                           DataSequence* X, Model* model,
                           int bComputePartition,int seqLabel,
                           bool bUseStatePerNodes);
    void computeBeliefsLinear(Beliefs& bel, FeatureGenerator* fGen,
                           DataSequence* X, Model* model,
                           int bComputePartition,int seqLabel,
                           bool bUseStatePerNodes);
};

//InferenceEngineFF: Forward-filter. This is an inference engine to be used
// for 'real-time' CRFs, CRFs which will receive continuous input.
//
//Observations are injected in a DataSequenceRealtime object, 
// one by one, until its buffer of length windowSize+delay is complete. Alphas/betas
// are propagated until/from the end of this buffer to compute beliefs for every frame (that is,
// once the buffer has been filled). Right now, inference won't be very optimized, but it is on the TODO list.
class InferenceEngineFF : public InferenceEngine
{
public:
	InferenceEngineFF(int inDelay);

	//Will compute a single belief vector for current position.
	int computeSingleBelief(FeatureGenerator* fgen, Model* model,
						DataSequenceRealtime* dataSeq, dVector* prob);

	//Compute beliefs for a complete datasequence, simulating real-time.
	//For now it will not store edge beliefs, only state beliefs.
    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,DataSequence* X,
                        Model* crf, int bComputePartition,int seqLabel=-1,
                        bool bUseStatePerNodes = false,bool bMaxProduct=false);

    double computePartition(FeatureGenerator* fGen, DataSequence* X,
                        Model* crf, int seqLabel=-1, 
						bool bUseStatePerNodes = false,bool bMaxProduct=false);

private:
    void computeBeliefsLog(Beliefs& bel, FeatureGenerator* fGen,
                           DataSequence* X, Model* model,
                           int bComputePartition,int seqLabel,
                           bool bUseStatePerNodes);

	int delay; //or filter length.
};

class InferenceEngineDummy:public InferenceEngine
{
  public:
    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,DataSequence* X, 
                       Model* crf, int bComputePartition,int seqLabel=-1, 
                       bool bUseStatePerNodes = false,bool bMaxProduct=false);
    double computePartition(FeatureGenerator* fGen, DataSequence* X, 
                        Model* crf,int seqLabel = -1, 
                        bool bUseStatePerNodes = false,bool bMaxProduct=false);
};

class InferenceEngineBrute:public InferenceEngine
{
  public:  
    InferenceEngineBrute();
    ~InferenceEngineBrute();

    void computeBeliefs(Beliefs& bel, FeatureGenerator* fGen, DataSequence* X, 
                       Model* crf, int bComputePartition, int seqLabel=-1, 
                       bool bUseStatePerNodes = false,bool bMaxProduct=false);
    double computePartition(FeatureGenerator* fGen, DataSequence* X, 
						Model* crf,int seqLabel=-1, 
						bool bUseStatePerNodes = false,bool bMaxProduct=false);
	int computeSingleBelief(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob);  
  private:
    int computeMaskedBeliefs(Beliefs& bel, FeatureGenerator* fGen,
                             DataSequence* X, Model* m, int bComputePartition,
                             int seqLabel=-1);
    double computeMaskedPartition(FeatureGenerator* fGen, DataSequence* X, 
                                  Model* m,int seqLabel=-1);
};

class InferenceEnginePerceptron// Similar to InferenceEngineFB, InferenceEnginePerceptron may be used for both CRF and LDCRF 
{
  public:    
    InferenceEnginePerceptron();
    ~InferenceEnginePerceptron();
   
	void computeViterbiPath(iVector& viterbiPath, FeatureGenerator* fGen,
                       DataSequence* X, Model* m,
                       int seqLabel = -1, bool bUseStatePerNodes = false);
	private:    
    void computeMi(FeatureGenerator* fGen, Model* model, DataSequence* X,
                      int index, int seqLabel, dMatrix& Mi_YY, dVector& Ri_Y,
                      bool bUseStatePerNodes);
	void ViterbiForwardMax(dMatrix& Mi_YY, dVector& Ri_Y, dVector& alpha_Y, 
		std::vector<iVector>& viterbiBacktrace,int index);	
};



class JTNode
{
public:
	JTNode();
	~JTNode();
	//
	void initialize(double initial_value=0.0);

	//
	dVector marginalize(int var, bool bMax);
	dVector marginalize(int var_a, int var_b, bool bMax);	
	dVector marginalize(std::list<int> sum_to, bool bMax);  

	//
	// Assign a singleton potential
	void assign_potential(int var, int state, double val);
	//
	// Assign a pairwise potential
	void assign_potential(int var_a, int var_b, int state_a, int state_b, double val);
	//
	void scale(std::list<int> vars, dVector ratio);

	// HELPER FUNCTIONS
	//
	void print_vars();
	void print_potentials();

	//
	bool equals(JTNode* node);
	bool contains(int var);
	bool contains(int var_a, int var_b);

	void sort();

	std::list<int> vars;
	std::list<int> cardinalities;
 
	int total_num_states;
	int* num_states;
	int** enum_states;
	dVector potentials; 
	dVector ratio; // for Separator node
};

class Clique;
class Separator;

class Clique: public JTNode
{
public: 
	Clique():JTNode() {}; 
	void add_neighbor(Separator* S, Clique* C);
	std::map<Clique*, Separator*> neighbors;
};

class Separator: public JTNode
{
public:
	Separator(Clique* node_A, Clique* node_B): 
	  JTNode(), clique_A(node_A), clique_B(node_B) {};
	void update_ratio(dVector new_potentials);
	Clique *clique_A, *clique_B;
};

class InferenceEngineJT: public InferenceEngine
{
public:
    InferenceEngineJT();
    ~InferenceEngineJT(); 
	
    void computeBeliefs(Beliefs& beliefs, FeatureGenerator* fGen, DataSequence* X,
		Model* model, int bComputePartition, int seqLabel=-1, 
		bool bUseStatePerNodes=false, bool bMaxProduct=false);

    double computePartition(FeatureGenerator* fGen, DataSequence* X, Model* model, 
		int seqLabel=-1, bool bUseStatePerNodes=false, bool bMaxProduct=false);
	
	int computeSingleBelief(FeatureGenerator* fGen, Model* model,
		DataSequenceRealtime* dataSequence, dVector* prob); 

private: 
	void constructTriangulatedGraph(
		std::vector<Clique*> &vecCliques, 
		Model* model, 
		DataSequence* X, 
		iMatrix adjMat,
		iVector nbStates);

	void buildJunctionTree(
		std::vector<Clique*> &vecCliques, 
		std::vector<Separator*> &vecSeparators,
		Model* model, 
		DataSequence* X, 
		iMatrix& JTadjMat,
		iVector nbStates);

	void removeRedundantNodes(
		std::vector<Clique*> &vecCliques,
		std::vector<Separator*> &vecSeparators,
		Model* model,
		DataSequence* X,		
		iMatrix& JTadjMat);

	// Helper functions for removeRedundantNodes()
	int findCliqueOffset(std::vector<Clique*> vecCliques, Clique* clique);
	int findSeparatorOffset(std::vector<Separator*> vecSeparator, Separator* separator);

	void findNodeToCliqueIndexMap(
		Model* model, 
		DataSequence* X, 
		std::vector<Clique*> vecCliques,  
		iVector& vecNodeToClique,
		iMatrix& matNodeToClique,
		iMatrix adjMat);

	void initializePontentials(
		std::vector<Clique*> &vecCliques, 
		std::vector<Separator*> &vecSeparators,
		FeatureGenerator* fGen, 
		Model* model, 
		DataSequence* X, 
		int seqLabel, 
		iMatrix adjMat, 		 
		iVector vecNodeToClique,
		iMatrix matNodeToClique,
		bool bUseStatePerNodes);
 
	void initializeBeliefs(
		Beliefs& beliefs, 
		DataSequence* X, 
		Model* m, 
		iMatrix adjMat,
		iVector nbStates);

	void updateBeliefs(
		std::vector<Clique*> vecCliques, 
		std::vector<Separator*> vecSeparators,
		Beliefs& beliefs, 
		Model* model, 
		DataSequence* X, 
		iMatrix adjMat,
		iVector nbStates,
		iVector vecNodeToClique,
		iMatrix matNodeToClique,
		bool bMaxProduct);

	void collectEvidence(
		std::vector<Clique*> vecCliques,  
		int node_idx, 
		iMatrix& untouched_nodes,
		bool bMaxProduct);
	
	void distributeEvidence(
		std::vector<Clique*> vecCliques,  
		int node_idx, 
		iMatrix& untouched_nodes,
		bool bMaxProduct);

	void update(
		Clique* src_clique, 
		Clique* dst_clique, 
		bool bMaxProduct);

	void printJunctionTree(
		std::vector<Clique*> vecCliques, 
		std::vector<Separator*> vecSeparators);

	void checkConsistency(
		std::vector<Clique*> vecCliques, 
		std::vector<Separator*> vecSeparators,
		bool bMaxProduct);

	int getNextElimination(iMatrix adjMat);
	std::list<int> getIntersect(Clique* a, Clique* b);

	void printBeliefs(Beliefs beliefs);
	
	//
	// Predicate for sorting JTCliques using std::sort(...)
	static bool CliqueSortPredicate(Clique* a, Clique* b) { 
		return (a->vars.front() < b->vars.front()); 
	}; 
};

class BPNode
{
public:
	BPNode(int node_id, int node_size);
	~BPNode();
	void addNeighbor(BPNode* v);
	bool equal(BPNode* v);
	void print();

	int id;
	int size;
	std::list<BPNode*> neighbors;
};
 

class InferenceEngineGBP: public InferenceEngine
{
public:
    InferenceEngineGBP();
    ~InferenceEngineGBP();  
	
    void computeBeliefs(Beliefs& beliefs, FeatureGenerator* fGen, DataSequence* X,
		Model* model, int bComputePartition, int seqLabel=-1, 
		bool bUseStatePerNodes=false,bool bMaxProduct=false);

    double computePartition(FeatureGenerator* fGen, DataSequence* X, Model* model, 
		int seqLabel=-1, bool bUseStatePerNodes=false,bool bMaxProduct=false);
	
	int computeSingleBelief(FeatureGenerator* fGen, Model* model,
		DataSequenceRealtime* dataSequence, dVector* prob); 
 
private: 
	void collect(BPNode* xi, BPNode* xj,
		dVector* psi_i, dMatrix** psi_ij, dVector** msg);
	
	void distribute(BPNode* xi, BPNode* xj,
		dVector* psi_i, dMatrix** psi_ij, dVector** msg);
	
	void sendMessage(BPNode* xi, BPNode* xj,
		dVector* psi_i, dMatrix** psi_ij, dVector** msg); 
	
	void initMessages(dVector**& msg, 
		DataSequence* X, Model* model, iMatrix adjMat, iVector nbStates);

	void initBeliefs(Beliefs& beliefs, 
		DataSequence* X, Model* model, iMatrix adjMat, iVector nbStates);	

	void initPotentials(dVector*& psi_i, dMatrix**& psi_ij, 
		FeatureGenerator* fGen, DataSequence* X, Model* model, 
		iMatrix adjMat, iVector nbStates, int seqLabel);
	
	void updateBeliefs(Beliefs& beliefs, 
		dVector* psi_i, dMatrix** psi_ij, dVector** msg,
		DataSequence* X, Model* model, iMatrix adjMat);	
 
	int logMultiplyMaxProd(dVector src_Vi, dMatrix src_Mij, dVector& dst_Vj); 
	
	void printBeliefs(Beliefs beliefs);   
};


class InferenceEngineLoopyBP: public InferenceEngine
{
public:
    InferenceEngineLoopyBP(int max_iter=15, double error_threshold=0.0001);
    ~InferenceEngineLoopyBP();  
	
    void computeBeliefs(Beliefs& beliefs, FeatureGenerator* fGen, DataSequence* X,
		Model* model, int bComputePartition, int seqLabel=-1, 
		bool bUseStatePerNodes=false,bool bMaxProduct=false);

    double computePartition(FeatureGenerator* fGen, DataSequence* X, Model* model, 
		int seqLabel=-1, bool bUseStatePerNodes=false,bool bMaxProduct=false);
	
	int computeSingleBelief(FeatureGenerator* fGen, Model* model,
		DataSequenceRealtime* dataSequence, dVector* prob); 
 
private: 
	// Convergence criterion
	int m_max_iter;
	double m_min_threshold;  

	void initializeBeliefs(Beliefs& beliefs, DataSequence* X, Model* m, iMatrix adjMat);
	void initializeMessages(std::vector<dVector>& messages, DataSequence* X, Model* m, iMatrix adjMat, int adjMatMax, bool randomInit = false);
	void initializePotentials(Beliefs& potentials, FeatureGenerator* fGen, DataSequence *X, Model* m, iMatrix adjMat, int seqLabel, bool bUseStatePerNodes=false);
	
	void normalizeMessage(int xi, std::vector<dVector>& messages, iMatrix adjMat, int adjMatMax);
 
	// m_ij(xj) = sum_xi {potential(i)*potential(i,j)*prod_{u \in N(i)\j} {m_ui(xi)}}
	// In case of maxProduct, returns an index of max_xi (for Viterbi decoding)
	void sendMessage(int xi, int xj, int numOfNodes, 
		const Beliefs potentials, 
		std::vector<dVector>& messages, 
		iMatrix adjMat,
		int adjMatMax,
		bool bMaxProd); 

	// b_i(xi) = potential(i) * prod_{u \in N(i)}{m_ui}
	// b_ij(xi,xj) = potential(i) * potential(j) * potential(i,j)
	//			     * prod_{u \in N(i)\j}{m_ui} * prod_{u \in N(j)\i}{m_uj}
	void updateBeliefs(int numOfNodes,
		Beliefs& beliefs, 
		const Beliefs potentials, 
		const std::vector<dVector> messages, 
		iMatrix adjMat,
		int adjMatMax );  

	//
	// Helper functions
	void getSequentialUpdateOrder(int* order, DataSequence* X, Model* m);
	void getRandomUpdateOrder(int* order, DataSequence* X, Model* m);
	void logMultiply(dVector src_Vi, dMatrix src_Mij, dVector& dst_Vj);
	int logMultiplyMaxProd(dVector src_Vi, dMatrix src_Mij, dVector& dst_Vj);
	void logMultiply(dVector src_Vi, dVector src_Vj, dMatrix& dst_Mij);
	
	void printBeliefs(Beliefs beliefs); 
	
	// For debugging purpose only. Computes exact marginals and partition
	double computePartitionBruteForce(FeatureGenerator* fGen, 
		DataSequence* X, Model* model, int seqLabel);
	double computeBeliefsBruteForce(Beliefs& beliefs, 
		FeatureGenerator* fGen, DataSequence* X, Model* model, int seqLabel);
	double evaluateLabelsBruteForce(std::vector<iVector> stateLabels, 
		FeatureGenerator* fGen, DataSequence* X, Model* model, int seqLabel);
};

class InferenceEnginePiecewise: public InferenceEngine
{
public:	
	int computeSingleBelief(FeatureGenerator* f,Model* m,DataSequenceRealtime* X,dVector* p){return 0;}; 	
    void computeBeliefs(Beliefs& b,FeatureGenerator* f,DataSequence* X,Model* m,int bCP,int y,bool bUSPN,bool bMP){};
    double computePartition(FeatureGenerator* f,DataSequence* X,Model* m,int y,bool bUSPN,bool bMP){return 0;};

	void computeBeliefs(Beliefs& beliefs, FeatureGenerator* fGen, DataSequence* X,
		Model* m, int seqLabel, int xi, int xj, iVector nbStates, bool bComputeMaxMargin); 

};

#endif //INFERENCEENGINE_H
