//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Model Component
//
//	January 19, 2006

#ifndef MODEL_H
#define MODEL_H

#define MAX_NUMBER_OF_LEVELS 10

//Standard Template Library includes
#include <vector>
#include <iostream>
#include "hcrfExcep.h"

//hCRF library includes
#include "dataset.h"

class FeatureGenerator;

enum eFeatureTypes
{
   allTypes = 0,
   edgeFeaturesOnly,
   nodeFeaturesOnly
};


// Type of graph topology
enum eGraphTypes {
   CHAIN,
   DANGLING_CHAIN, // chain of hidden states with attached labels
   MV_GRAPH_LINKED,			// MV_: prefix for multi-view models
   MV_GRAPH_COUPLED,
   MV_GRAPH_LINKED_COUPLED,
   MV_GRAPH_PREDEFINED,
   ADJMAT_PREDEFINED
};

enum{
   ALLSTATES,
   STATES_BASED_ON_LABELS,
   STATEMAT_PREDEFINED,
   STATEMAT_PROBABILISTIC
};

//-------------------------------------------------------------
// Model Class
//

class Model {
public:
   Model(int numberOfStates = 0, int numberOfSeqLabels = 0,
         int numberOfStateLabels = 0);
   ~Model();
   void setAdjacencyMatType(eGraphTypes atype, ...);
   eGraphTypes getAdjacencyMatType();
   int setStateMatType(int stype, ...);
   int getStateMatType();
// adjacency and state matrix sizes are max-sizes, based on the
// longest sequences seen thus far; use sequences length instead for
// width and height of these matrices
   void getAdjacencyMatrix(uMatrix&, DataSequence* seq);
   iMatrix * getStateMatrix(DataSequence* seq);
   iVector * getStateMatrix(DataSequence* seq, int nodeIndex);

   void setWeights(const dVector& weights);
   dVector * getWeights(int seqLabel = -1);
   void refreshWeights();

   int getNumberOfStates() const;
   void setNumberOfStates(int numberOfStates);

   int getNumberOfStateLabels() const;
   void setNumberOfStateLabels(int numberOfStateLabels);

   int getNumberOfSequenceLabels() const;
   void setNumberOfSequenceLabels(int numberOfSequenceLabels);
   
   int getNumberOfRawFeaturesPerFrame();
   void setNumberOfRawFeaturesPerFrame(int numberOfRawFeaturesPerFrame);

   void setRegL1Sigma(double sigma, eFeatureTypes typeFeature = allTypes);
   void setRegL2Sigma(double sigma, eFeatureTypes typeFeature = allTypes);
   double getRegL1Sigma();
   double getRegL2Sigma();
   eFeatureTypes getRegL1FeatureTypes();
   eFeatureTypes getRegL2FeatureTypes();
   
   //Alpĥa regularization used in LDCNF. I have not found a simpler way to make it accessible for gradient computation,
   //so here it is.
   void setAlphaRegL1(double inAlphaRegL1);
   double getAlphaRegL1();
   void setAlphaRegL2(double inAlphaRegL2);
   double getAlphaRegL2();
   
   void setFeatureMask(iMatrix &ftrMask);
   iMatrix* getFeatureMask();
   int getNumberOfFeaturesPerLabel();

   iMatrix& getStatesPerLabel();
   iVector& getLabelPerState();
   int getDebugLevel();
   void setDebugLevel(int newDebugLevel);

   void load(const char* pFilename);
   void save(const char* pFilename) const;

   int read(std::istream* stream);
   int write(std::ostream* stream) const;

   uMatrix* getInternalAdjencyMatrix();
   iMatrix *getInternalStateMatrix();
   
   // Multi-View Support
   Model(int numberOfViews, int* numberOfStatesMultiView,
	   int numberOfSeqLabels = 0, int numberOfStateLabels = 0); 
   
   bool isMultiViewMode() const;

   void setNumberOfViews(int numberOfViews);
   int getNumberOfViews() const;
    
   void setNumberOfStatesMV(int* numberOfStatesMultiView);  
   int getNumberOfStatesMV(int view) const;

   iMatrix& getStatesPerLabelMV(int view);
   iVector& getLabelPerStateMV(int view);

   void setRawFeatureIndexMV(std::vector<std::vector<int> > rawFeatureIndexPerView);
   std::vector<int> getRawFeatureIndexMV(int view) const; 

   void getAdjacencyMatrixMV(iMatrix&, DataSequence* seq);

   // 
   void useMaxMargin(bool useMaxMargin) {bComputeMaxMargin = useMaxMargin;};
   void useNRBM(bool useNRBM) {bUseNRBM = useNRBM;};
   bool isMaxMargin() {return bComputeMaxMargin;};
   bool isNRBM() {return bUseNRBM;};

private:
   int numberOfSequenceLabels;
   int numberOfStates;
   int numberOfStateLabels;
   int numberOfFeaturesPerLabel;   
   int numberOfRawFeaturesPerFrame;

   dVector weights;
   std::vector<dVector> weights_y;

   double regL1Sigma;
   double regL2Sigma;
   eFeatureTypes regL1FeatureType;
   eFeatureTypes regL2FeatureType;

   double alphaRegL1;
   double alphaRegL2;

   int debugLevel;

   eGraphTypes adjMatType;
   int stateMatType;
   uMatrix adjMat;
   iMatrix stateMat, featureMask;
   iMatrix statesPerLabel;
   iVector stateVec, labelPerState;

   int loadAdjacencyMatrix(const char *pFilename);
   int loadStateMatrix(const char *pFilename);

   void makeChain(uMatrix& m, int n);
   void predefAdjMat(uMatrix& m, int n);
   iMatrix * makeFullStateMat(int n);
   iMatrix * makeLabelsBasedStateMat(DataSequence* seq);
   iMatrix * predefStateMat(int n);
   void updateStatesPerLabel(); 
   
   // Max margin    
   bool bComputeMaxMargin;
   bool bUseNRBM;

   // Multi-View Support
   int numberOfViews;
   int* numberOfStatesMV; 
   std::vector<std::vector<int> > rawFeatureIndex;
   std::vector<iVector> labelPerStateMV;
   std::vector<iMatrix> statesPerLabelMV;

   // Creates adjMat for multiview chains. Contains unique edgeID.
   void makeChainMV(iMatrix& m, int seqLength);    
   void updateStatesPerLabelMV(); 
};

// stream io routines
std::istream& operator >>(std::istream& in, Model& m);
std::ostream& operator <<(std::ostream& out, const Model& m);

#endif
