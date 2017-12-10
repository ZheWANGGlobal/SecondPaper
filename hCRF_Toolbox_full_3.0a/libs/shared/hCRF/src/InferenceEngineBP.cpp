#include "inferenceengine.h"
#ifdef _OPENMP
#include <omp.h>
#endif

//*
// Constructor and Destructor
//*

InferenceEngineBP::InferenceEngineBP() : InferenceEngine()
{
}

InferenceEngineBP::InferenceEngineBP(const InferenceEngineBP& other)
{
	this->operator=(other);
}

InferenceEngineBP&  InferenceEngineBP::operator=(const InferenceEngineBP& other)
{
    AdjacencyMatrix = other.AdjacencyMatrix;
	theta = new dVector(*other.theta);
	return *this;
}

InferenceEngineBP::~InferenceEngineBP()
{
   //does nothing
}

//*
// Public Methods
//*

// Inference Functions
int InferenceEngineBP::computeSingleBelief(FeatureGenerator* fGen, Model* model,
						DataSequenceRealtime* dataSequence, dVector* prob)
{
	return 0;
}
void InferenceEngineBP::computeBeliefs(Beliefs& bel, FeatureGenerator* fGen,
									  DataSequence* X, Model* crf,
									  int bComputePartition, int seqLabel,
									  bool bUseStatePerNodes, bool bMaxProduct)
{
	if( bMaxProduct )
		throw HcrfNotImplemented("InferenceEngineBP for max-margin is not implemented");

	int i,j,k;
	NSTATES=crf->getNumberOfStates();
	NNODES=X->length(); //ask if this is the number of nodes
	NFEATURES=fGen->getNumberOfFeatures(allTypes,seqLabel);
	crf->getAdjacencyMatrix(AdjacencyMatrix, X);
	theta=crf->getWeights(seqLabel);
	NEDGES=CountEdges(AdjacencyMatrix, NNODES);
	Tree.create(NNODES,NNODES);
//	Tree.set(0);
	iVector PostOrder;
	iVector PreOrder;
	PostOrder.create(NNODES, COLVECTOR);
	PreOrder.create(NNODES, COLVECTOR);
//	PostOrder.set(0);
//	PreOrder.set(0);
	dfs(PreOrder,PostOrder);
	BuildEdgeMatrix2();
	bel.belStates.resize(NNODES);
	for(i=0;i<NNODES;i++)
	{
		bel.belStates[i].create(NSTATES);
//		bel.belStates[i].set(0);
	}
	bel.belEdges.resize(NEDGES*2);
	for(i=0;i<NEDGES*2;i++)
	{
		bel.belEdges[i].create(NSTATES,NSTATES);
//		bel.belEdges[i].set(0);
	}


	TreeInfer(bel,fGen,X,PostOrder,PreOrder,crf,seqLabel,bUseStatePerNodes);

	for (i=0;i<NNODES;i++)
	{
		for(j=0;j<NSTATES;j++){
			bel.belStates[i].setValue(j,exp(bel.belStates[i].getValue(j)));
		}
//        printf(" Beliefs at node %d \n",i);
//		bel.belStates[i].display();
	}

	for(i=0;i<NEDGES;i++){ // ask LUI i start at 1 and not zero
		for(j=0;j<NSTATES;j++){
			for(k=0;k<NSTATES;k++){
				bel.belEdges[i].setValue(j, k, exp(bel.belEdges[i].getValue(j,k)));
			}
		}
//        printf(" Beliefs at ed %d \n",i);
//		bel.belEdges[i].display();
	}
	if(bComputePartition){
		bel.partition = computePartition(fGen,X,crf,seqLabel,bUseStatePerNodes);
        // TODO: Optimize code so we can pass (PostOrder,PreOrder)
	}
}


double InferenceEngineBP::computePartition(FeatureGenerator* fGen, 
										   DataSequence* X, Model* crf,
										   int seqLabel, bool bUseStatePerNodes, bool bMaxProduct)
{
	if( bMaxProduct )
		throw HcrfNotImplemented("InferenceEngineBP for max-margin is not implemented");

	int i; int j; int t; int k;
	NSTATES=crf->getNumberOfStates();
	NNODES=X->length(); //ask if this is the number of nodes
	NFEATURES=fGen->getNumberOfFeatures(allTypes,seqLabel);
	crf->getAdjacencyMatrix(AdjacencyMatrix, X);
	theta=crf->getWeights(seqLabel);
	NEDGES=CountEdges(AdjacencyMatrix, NNODES);
	Tree.create(NNODES,NNODES);
	int nodeI,par;
	int edgeNum;
	int stateNumI;
	int stateNumJ;
	int childI;
	int parentI;
	double value;
	int NumberOfChildren;
	int NumberOfParents;
	int p1; //,p3;
	BuildEdgeMatrix2();
	p1=0;
	dVector newm(NSTATES);
	dVector lastm(NSTATES);
	dVector vectorI(NSTATES);
	dVector tempm(NSTATES);
	dVector m(NSTATES);
	iVector ch(NNODES);
	iVector pa(NNODES);
	dVector VectorEntry;
	dMatrix Msg(NSTATES,1);
	dMatrix LogPotij(NSTATES,NSTATES);
	dVector vectorj(NSTATES);
	iVector PostOrder;
	iVector PreOrder;
	PostOrder.create(NNODES, COLVECTOR);
	PreOrder.create(NNODES, COLVECTOR);
	dfs(PreOrder,PostOrder);

	Beliefs bel;

	bel.belStates.resize(NNODES);
	for(i=0;i<NNODES;i++)
	{
		bel.belStates[i].create(NSTATES);
	}
	bel.belEdges.resize(NEDGES*2);
	for(i=0;i<NEDGES*2;i++)
	{
		bel.belEdges[i].create(NSTATES,NSTATES);
	}

  dMatrix LogLocalEvidence;
	dMatrix MatrixEntry;
	std::vector<dMatrix> LogEdgeEvidence;
	std::vector<dMatrix> Messages; //n nodes by nnodes

	// Step 1: Get LocalEvidence and EdgeEvidence
	LogLocalEvidence.create(NSTATES,NNODES);
	MakeLocalEvidence(LogLocalEvidence,fGen,X,crf,seqLabel);
	MakeEdgeEvidence(LogEdgeEvidence,fGen,X,crf,seqLabel);
   //Step 1b: If bUseStatePerNodes== TRUE, mask certain states based on X->getStatesPerNode()
   if(bUseStatePerNodes)
   {
		iMatrix* pStatesPerNodes = crf->getStateMatrix(X);
		for(int n = 0; n < NNODES;n++)
		{
			for(int s = 0; s < NSTATES; s++)
			{
				if(pStatesPerNodes->getValue(s,n) == 0)
					LogLocalEvidence.setValue(n,s,-INF_VALUE);
			}
		}
   }

  // Step 2: Initialize messages to zero for every edge
   Messages.resize(NSTATES);
  for(stateNumI=0;stateNumI<NSTATES;stateNumI++)
  {
	  Messages[stateNumI].create(NNODES,NNODES); //nodes by nodes
  }
// Step 3: Collect to root
//   printf("Step 3 : Collect to root \n");
  for(t=0;t<NNODES;t++){
	  i=PostOrder.getValue(t);
	  ch.create(NNODES, COLVECTOR);
	  pa.create(NNODES, COLVECTOR);
	  ch.set(-1);
	  pa.set(-1);
      NumberOfChildren=children(i,ch,Tree);
      NumberOfParents=parents(i,pa,Tree);
      for(stateNumI=0;stateNumI<NSTATES;stateNumI++){
		  bel.belStates[i].setValue(stateNumI,LogLocalEvidence.getValue(i,stateNumI));
	  }
    // Step 4: Propagate Belief from children
      if(NumberOfChildren>0){
    // For all the children
        for(childI=0;childI<NumberOfChildren;childI++){
		    k=ch.getValue(childI);
            for(stateNumI=0;stateNumI<NSTATES;stateNumI++){
				value = bel.belStates[i].getValue(stateNumI) 
					+ Messages[stateNumI].getValue(k,i);
				bel.belStates[i].setValue(stateNumI,value);
		    }

		 }// end for all children
      } // if there there are children
    // if there is a parent

    if(NumberOfParents>0)
	{
        for(parentI=0;parentI<NumberOfParents;parentI++)
		{
			j=pa.getValue(parentI);
            LogPotij.create(NSTATES,NSTATES);
            GetEdgeMatrix(i,j,LogPotij,LogEdgeEvidence);
			newm.create(NSTATES);
	        tempm.create(NSTATES);
	        LogMultiply(LogPotij,bel.belStates[i],newm);
            for(stateNumI=0;stateNumI<NSTATES;stateNumI++)
			{
		        Messages[stateNumI].setValue(i,j,newm.getValue(stateNumI));
            }

		}// end for all parents
	}// End if there are parents


  }// end for all the nodes
   // distribute from root

  for(t=0;t<NNODES;t++)
  {
	  i=PreOrder.getValue(t);
	  NumberOfChildren=children(i,ch,Tree);
	  NumberOfParents=parents(i,pa,Tree);
	  if(NumberOfParents>0)
	  {
		  for(stateNumI=0;stateNumI<NSTATES;stateNumI++)
		  {
			  par=pa.getValue(0);
			  value=bel.belStates[i].getValue(stateNumI) + 
				  Messages[stateNumI].getValue(par,i);
			  bel.belStates[i].setValue(stateNumI,value);
		  }
	  }
	  // Pass down to children (if any)
	  for(nodeI=0;nodeI<NumberOfChildren;nodeI++)
	  {
		  j=ch.getValue(nodeI);
		  LogPotij.create(NSTATES,NSTATES);
		  edgeNum=GetEdgeMatrix(i,j,LogPotij,LogEdgeEvidence); // j,i
		  m.create(NSTATES);
		  for(stateNumJ=0;stateNumJ<NSTATES;stateNumJ++){
			  value=Messages[stateNumJ].getValue(j,i);
			  m.setValue(stateNumJ,value);
			  //if(isinf(m.getValue(stateNumJ)))
			  //INT_MIN
			  if(m.getValue(stateNumJ)==INT_MAX){ 
                   // bug in the math.h library this should be -1 log(0)=-Inf
				  value=m.getValue(stateNumJ)+1;
				  m.setValue(stateNumJ,value);
			  }
		  }
		  tempm.create(NSTATES);
	      for(stateNumJ=0;stateNumJ<NSTATES;stateNumJ++){
			  value=bel.belStates[i].getValue(stateNumJ) - m.getValue(stateNumJ);
			  tempm.setValue(stateNumJ,value);
		  }
		  
		  newm.create(NSTATES);
	      LogMultiply(LogPotij,tempm, newm);
		  lastm=newm;
          for(stateNumJ=0;stateNumJ<NSTATES;stateNumJ++){
			  Messages[stateNumJ].setValue(i,j,newm.getValue(stateNumJ));
		  }
	  }// end for all children
  }// end for all nodes
  dVector Ztemp;
  Ztemp.create(NSTATES);
  // TODO: Figure out what j is here. It may be uninitalised. It is unclear what
  // this compute
  for(stateNumI=0;stateNumI<NSTATES;stateNumI++)
  {
	  value=LogLocalEvidence.getValue(j,stateNumI)+lastm.getValue(stateNumI);
	  Ztemp.setValue(stateNumI,value);
  }
  double Z,m1,m2,sub;
  int col;
  
  Z=Ztemp.getValue(0);
  for(col=1;col<NSTATES;col++){
	  if(Z >= Ztemp.getValue(col)){
		  m1=Z;
		  m2=Ztemp.getValue(col);
	  } else {
		  m1=Ztemp.getValue(col);
		  m2=Z;
	  }
  	  sub=m2-m1;
	  Z=m1 + log(1 + exp(sub));
  }
  return Z;
}

void InferenceEngineBP::BuildEdgeMatrix2()
{
	int nodeI,nodeJ;
	int index=1;
	EMatrix.create(NNODES,NNODES);
//	EMatrix.set(0);
	for (nodeI=0;nodeI<NNODES;nodeI++){
		for(nodeJ=0;nodeJ<NNODES;nodeJ++){
			if(AdjacencyMatrix(nodeI,nodeJ) ==1 && nodeI<nodeJ)
			{
				EMatrix.setValue(nodeI,nodeJ,index);
				index=index+1;
			}

		}
	}
}

//*
// Private Methods
//*

// This Function returns a dMatrix(i,j)=local evidence for node i state j
void InferenceEngineBP::MakeLocalEvidence(dMatrix& LogLocalEvidence, FeatureGenerator* fGen, DataSequence* X, Model* crf, int seqLabel)
{

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

	for(int n = 0; n < NNODES ; n++)
	{
#if defined(_VEC_FEATURES) || defined(_OPENMP)
		fGen->getFeatures(vecFeaturesMP[ThreadID], X, crf, n, -1, seqLabel);
		feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
		for(int j = 0; j < vecFeaturesMP[ThreadID].size(); j++, pFeature++){
#else
		vecFeatures = fGen->getFeatures(X,crf,n,-1,seqLabel);
		feature* pFeature = vecFeatures->getPtr();
		for(int j = 0; j < vecFeatures->size(); j++, pFeature++) {
#endif
			LogLocalEvidence(n,pFeature->nodeState) += pFeature->value * theta->getValue(pFeature->id);
		}
	}
}

void InferenceEngineBP::MakeEdgeEvidence(std::vector<dMatrix>& LogEdgeEvidence,FeatureGenerator* fGen,DataSequence* X, Model* crf, int seqLabel)
{
	// need to get edgenum as parameter
	int nodeI;
	int nodeJ;
	int i;
	int EdgeNum=0;
	LogEdgeEvidence.resize(NEDGES);
	for(i=0;i<NEDGES;i++) {
		LogEdgeEvidence[i].create(NSTATES,NSTATES);
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
	for (nodeI=0;nodeI<NNODES;nodeI++) {
		for (nodeJ=0;nodeJ<NNODES;nodeJ++) {
			if(nodeI<nodeJ && AdjacencyMatrix(nodeI,nodeJ) == 1) {
#if defined(_VEC_FEATURES) || defined(_OPENMP)
				fGen->getFeatures(vecFeaturesMP[ThreadID], X,crf,nodeJ,nodeI,seqLabel);
				feature* pFeature = vecFeaturesMP[ThreadID].getPtr();
				for(int j = 0; j < vecFeaturesMP[ThreadID].size(); j++, pFeature++) {
#else
				vecFeatures = fGen->getFeatures(X,crf,nodeJ,nodeI,seqLabel);
				feature* pFeature = vecFeatures->getPtr();
				for(int j = 0; j < vecFeatures->size(); j++, pFeature++) {
#endif
					LogEdgeEvidence[EdgeNum](pFeature->prevNodeState,pFeature->nodeState) += pFeature->value * theta->getValue(pFeature->id);
				}
				EdgeNum++;
			}
		}
	}
}

int InferenceEngineBP::GetEdgeMatrix(int nodeI, int nodeJ, dMatrix& EvidenceEdgeIJ, std::vector<dMatrix>& EdgePotentials)
{
	int EdgeNum = 0;
	if(nodeI > nodeJ)
	{
		EdgeNum = EMatrix(nodeJ,nodeI) - 1;
		EvidenceEdgeIJ.set(EdgePotentials[EdgeNum]);
	}
	else
	{
		EdgeNum = EMatrix(nodeI,nodeJ) - 1;
		Transpose(EdgePotentials[EdgeNum],EvidenceEdgeIJ);
	}

	return EdgeNum;

}

// This function returns P(i=a) for every node and every state, and P(i=a,j=b) for every edge and pair of states
void InferenceEngineBP::TreeInfer(Beliefs& bel,FeatureGenerator* fGen,DataSequence* X,iVector &PostOrder, iVector &PreOrder,Model* crf, int seqLabel, bool bUseStatePerNodes)
{
  int i;int j;int t;int k;//int loopC3;int pindex;
  int nodeI,par;
  int edgeNum;
  int stateNumI;
  int stateNumJ;
  int childI;
  int parentI;
  double value;
  int NumberOfChildren;
  int NumberOfParents;
  int p1;//,p3;
  int Max=0; // this to indicate map or viterbi
  p1=0;
  dVector newm(NSTATES);
  dVector lastm(NSTATES);
  dVector vectorI(NSTATES);
  dVector tempm(NSTATES);
  dVector m(NSTATES);
  iVector ch(NNODES);
  iVector pa(NNODES);
  dVector VectorEntry;
  dMatrix Msg(NSTATES,1);
  dMatrix LogPotij(NSTATES,NSTATES);
  dVector vectorj(NSTATES);

  dMatrix LogLocalEvidence;
  dMatrix MatrixEntry;
  std::vector<dMatrix> LogEdgeEvidence;
  std::vector<dMatrix> Messages; //n nodes by nnodes

   // Step 1: Get LocalEvidence and EdgeEvidence
   LogLocalEvidence.create(NSTATES,NNODES);
//   LogLocalEvidence.set(0);
   MakeLocalEvidence(LogLocalEvidence,fGen,X,crf,seqLabel);
   MakeEdgeEvidence(LogEdgeEvidence,fGen,X,crf, seqLabel);
   //Step 1b: If bUseStatePerNodes== TRUE, mask certain states based on X->getStatesPerNode()
   if(bUseStatePerNodes)
   {
		iMatrix* pStatesPerNodes = crf->getStateMatrix(X);
		for(int n = 0; n < NNODES;n++)
		{
			for(int s = 0; s < NSTATES; s++)
			{
				if(pStatesPerNodes->getValue(s,n) == 0)
					LogLocalEvidence.setValue(n,s,-INF_VALUE);
			}
		}
   }
  // Step 2: Initialize messages to zero for every edge
   Messages.resize(NSTATES);
  for(stateNumI=0;stateNumI<NSTATES;stateNumI++)
  {
	  Messages[stateNumI].create(NNODES,NNODES); //nodes by nodes
//	  Messages[stateNumI].set(0);
  }

  // Initialize belStates

// Step 3: Collect to root
//   printf("Step 3 : Collect to root \n");

  for(t=0;t<NNODES;t++){
	  i=PostOrder.getValue(t);
	  ch.create(NNODES);
	  pa.create(NNODES);
	  ch.set(-1);
	  pa.set(-1);
      NumberOfChildren=children(i,ch,Tree);
      NumberOfParents=parents(i,pa,Tree);
      for(stateNumI=0;stateNumI<NSTATES;stateNumI++){
		  bel.belStates[i].setValue(stateNumI,LogLocalEvidence.getValue(i,stateNumI));
	  }
    // Step 4: Propagate Belief from children
      if(NumberOfChildren>0){
    // For all the children
        for(childI=0;childI<NumberOfChildren;childI++){
		    k=ch.getValue(childI);
            for(stateNumI=0;stateNumI<NSTATES;stateNumI++){
				value=bel.belStates[i].getValue(stateNumI) + Messages[stateNumI].getValue(k,i);
				bel.belStates[i].setValue(stateNumI,value);
		    }

		 }// end for all children
      } // if there there are children
    // if there is a parent

    if(NumberOfParents>0){
        for(parentI=0;parentI<NumberOfParents;parentI++){
			j=pa.getValue(parentI);
            LogPotij.create(NSTATES,NSTATES);
            GetEdgeMatrix(i,j,LogPotij,LogEdgeEvidence);
			newm.create(NSTATES, COLVECTOR);
	        tempm.create(NSTATES,COLVECTOR);
            if (Max){
	                // change for Viterbi
                    // newm = max_mult(pot_ij, bel{i});
             }

            if(Max==0){
	             LogMultiply(LogPotij,bel.belStates[i],newm);
             }
			LogNormalise(newm, tempm);
            for(stateNumI=0;stateNumI<NSTATES;stateNumI++){
                newm.setValue(stateNumI,tempm.getValue(stateNumI));
            }
            for(stateNumI=0;stateNumI<NSTATES;stateNumI++){
		        Messages[stateNumI].setValue(i,j,newm.getValue(stateNumI));
            }

    }// end for all parents
	}// End if there are parents


  }// end for all the nodes
   // distribute from root

    for(t=0;t<NNODES;t++){
          i=PreOrder.getValue(t);
          NumberOfChildren=children(i,ch,Tree);
          NumberOfParents=parents(i,pa,Tree);
          if(NumberOfParents>0){
             for(stateNumI=0;stateNumI<NSTATES;stateNumI++){
				 par=pa.getValue(0);
				 value=bel.belStates[i].getValue(stateNumI)+ Messages[stateNumI].getValue(par,i);
				 bel.belStates[i].setValue(stateNumI,value);
			 }
		  }
		  tempm.create(NSTATES, COLVECTOR);
		  LogNormalise(bel.belStates[i], tempm);
          for(stateNumI=0;stateNumI<NSTATES;stateNumI++){
			  bel.belStates[i].setValue(stateNumI,tempm.getValue(stateNumI));
              }
        // Pass down to children (if any)
	     for(nodeI=0;nodeI<NumberOfChildren;nodeI++){
		     j=ch.getValue(nodeI);
		     LogPotij.create(NSTATES,NSTATES);
             edgeNum=GetEdgeMatrix(i,j,LogPotij,LogEdgeEvidence); // j,i
		     m.create(NSTATES, COLVECTOR);
             for(stateNumJ=0;stateNumJ<NSTATES;stateNumJ++){
				  value=Messages[stateNumJ].getValue(j,i);
				  m.setValue(stateNumJ,value);
				//if(isinf(m.getValue(stateNumJ)))
				//INT_MIN
                if(m.getValue(stateNumJ)==INT_MAX){ // bug in the math.h library this should be -1 log(0)=-Inf
					value=m.getValue(stateNumJ)+1;
					m.setValue(stateNumJ,value);
	            }
		   }
		  tempm.create(NSTATES, COLVECTOR);
	      for(stateNumJ=0;stateNumJ<NSTATES;stateNumJ++){
				value=bel.belStates[i].getValue(stateNumJ) - m.getValue(stateNumJ);
                tempm.setValue(stateNumJ,value);
			}

	        if(Max==1){
	           //  newm = max_mult(pot_ij, tmp);
	        }
            if(Max==0){
				newm.create(NSTATES, COLVECTOR);
	            LogMultiply(LogPotij,tempm, newm);
				lastm=newm;
				tempm.create(NSTATES, COLVECTOR);
                LogNormalise(newm, tempm);
                for(stateNumJ=0;stateNumJ<NSTATES;stateNumJ++){
                    newm.setValue(stateNumJ,tempm.getValue(stateNumJ));
                }
			}
            for(stateNumJ=0;stateNumJ<NSTATES;stateNumJ++){
				 Messages[stateNumJ].setValue(i,j,newm.getValue(stateNumJ));
			}

	  }// end for all children
    }// end for all nodes
	computeEdgeBel(bel,Messages,LogEdgeEvidence);

}

double InferenceEngineBP::computeZ(int, int, dVector&, 
								   dMatrix&)
{
	throw HcrfNotImplemented("InferenceEngineBP::computeZ");
}


// This function returns P(i=a,j=b) for all edges and all states
int InferenceEngineBP::computeEdgeBel(Beliefs& bel,std::vector<dMatrix> Messages, std::vector<dMatrix>& LogEdgeEvidence)
{
	int nnbrs,i,j,stateI,stateJ,e,nn;
	double value;
	iVector NBRS;
    NBRS.create(NNODES);
	dMatrix EvidenceEdgeIJ;
	EvidenceEdgeIJ.create(NSTATES,NSTATES);
    dMatrix T;
	T.create(NSTATES,NSTATES);
	dVector beli;
	beli.create(NSTATES);
	dVector belj;
	belj.create(NSTATES);
	dMatrix Result1;
	dMatrix Result2;
	dMatrix Result3;
	dMatrix Result4;

	Result1.create(NSTATES,NSTATES);
	Result2.create(NSTATES,NSTATES);
	Result3.create(NSTATES,NSTATES);
	Result4.create(NSTATES,NSTATES);

	for(i=0;i<NNODES;i++)
	{
	    NBRS.set(-1);
        nnbrs=findNbrs(i,NBRS);
	    for(nn=0;nn<nnbrs;nn++){
		   j=NBRS.getValue(nn);
		   EvidenceEdgeIJ.set(0);
           GetEdgeMatrix(i,j,EvidenceEdgeIJ,LogEdgeEvidence);
		   if(j>i){
			  T.set(0);
			  Transpose(EvidenceEdgeIJ,T);
              EvidenceEdgeIJ=T;
		   }
           for(stateI=0;stateI<NSTATES;stateI++){
			   value=bel.belStates[i].getValue(stateI) - Messages[stateI].getValue(j,i);
			   beli.setValue(stateI,value);
		   }

		   for(stateI=0;stateI<NSTATES;stateI++){
			   value=bel.belStates[j].getValue(stateI) - Messages[stateI].getValue(i,j);
			   belj.setValue(stateI,value);
		   }
		    repMat1(beli,NSTATES,Result1);
		    repMat2(belj,NSTATES,Result2);
		    for(stateI=0;stateI<Result1.getWidth();stateI++){
				for(stateJ=0;stateJ<Result1.getHeight();stateJ++){
					value=Result1.getValue(stateI,stateJ)+ Result2.getValue(stateI,stateJ);
					value=value + EvidenceEdgeIJ.getValue(stateI,stateJ);
					Result3.setValue(stateI,stateJ,value);
				}
			}
			Result4.set(0);
            LogNormalise2(Result3,Result4);
          	e=EMatrix.getValue(i,j);
			if(e>0){
			   for(stateI=0;stateI<Result1.getHeight();stateI++){
				  for(stateJ=0;stateJ<Result1.getWidth();stateJ++){
					  bel.belEdges[e-1].setValue(stateI,stateJ,Result4.getValue(stateI,stateJ));
				  }
			   }
			}


       }
	}

	return 0;
  }


void InferenceEngineBP::repMat1(dVector& beli,int nr, dMatrix& Result)
{
	/*
	This function take as input a vector of belief(length should be NSTSATES) and an integer (nr). The
	result is a Matrix containing nr copy of belief
	*/
	double value;
	Result.create(NSTATES,nr);
//	Result.set(0);
	for(int i=0;i<NSTATES;i++){
		value=beli.getValue(i);
		for(int j=0;j<nr;j++){
		    Result.setValue(i,j,value);
		}
	}

}

//TODO: Check the dimension. There seems to be a bug here if nr!= NSTATES
void InferenceEngineBP::repMat2(dVector& beli,int nr, dMatrix& Result)
{
	/* Same as repMat2, just return the transposed matrix
	*/
	int i,j;
	double value;
	Result.create(NSTATES,nr);
//	Result.set(0);
	for(i=0;i<NSTATES;i++){
		value=beli.getValue(i);
		for(j=0;j<nr;j++){
		    Result.setValue(j,i,value);
		}
	}

}

// Given and adjacency dMatrix this function computes the set of children of a
//node i and returns the number of children 

//TODO: Why return -1 if no children ( and not 0) 

//TODO: THere can be a bug if Child is not big enough to hold all the children
int InferenceEngineBP::children(int NodeNumber, iVector& Child, uMatrix& Ad)
{
   int i;
   int numOfChildren=0;
   
   for(i=0;i<NNODES;i++){
	  if(Ad.getValue(NodeNumber,i)==1){
		 Child.setValue(numOfChildren,i);
		 numOfChildren=numOfChildren+1;
	  }
   }
   if(numOfChildren > 0)
	  return numOfChildren;
   return -1;
}

// Same comment that in children
int InferenceEngineBP::parents(int NodeNumber,iVector& Parent,uMatrix& Ad)
{
	int i;
    int index=0;

    for(i=0;i<NNODES;i++){
	   if(Ad.getValue(i,NodeNumber)==1){
          Parent.setValue(index,i);
          index=index+1;
	   }
    }
	if(index > 0)
	   return index;
	return -1;
}

// Given and adjecency dMatrix this function returns the neighbors of a node,
// i.e. the union of children and parent nodes

// TODO: This function is bugy, it only return the parents. (It is also never
// used)
int InferenceEngineBP::neighboor(int NodeNumber,iVector& Neighboors, uMatrix& Ad)
{
	throw HcrfNotImplemented("InferenceEngineBP::neighboor is buggy");
	iVector Parent(NNODES);// max number of nodes
    iVector Child(NNODES); // max number of nodes
	Parent.set(-1);
	Child.set(-1);
//    int NumChildren;
//    int NumParents;
    int NumNeighboors;
	Neighboors.set(-1);
	NumNeighboors= parents(NodeNumber,Neighboors,Ad);
    return NumNeighboors;
}

// Given and adjacency dMatrix ( assumed to be a tree) this function returns the
// forward and backword ordering for msg passing
int InferenceEngineBP::dfs(iVector& Pred, iVector& Post)
{
  int i;

  white_global = 0;
  gray_global = 1;
  black_global = 2;
  color_global.create(NNODES);
//  color_global.set(0);
  time_stamp_global=0;

  d_global.create(NNODES);
//  d_global.set(0);
  f_global.create(NNODES);
//  f_global.set(0);
  pred_global.create(NNODES);
  pred_global.set(-1);

  cycle_global=0;

  pre_global.create(NNODES);
  post_global.create(NNODES);

  time_stamp_global = 0;
  cycle_global = 0;
  sizepre_global=0;
  sizepost_global=0;

  for(i=0;i<NNODES;i++){
	  if (color_global.getValue(i)==white_global){
		  Dfs_Visit(i, AdjacencyMatrix);
      }
   }
  for(i=0;i<NNODES;i++){
	  Pred.setValue(i,pre_global.getValue(i));
	  Post.setValue(i,post_global.getValue(i));
  }
  // Initialize tree
  for (i=0;i<NNODES;i++){
	  if (pred_global.getValue(i)>-1){
		  Tree.setValue(pred_global.getValue(i),i,1);
	  }
  }
  return 0 ; // size of pred_global and post_global must be equal to the number
			 // of nodes
}

// This is a utility function for dfs
void InferenceEngineBP::Dfs_Visit(int u,uMatrix& Ad)
{
  int i;

  pre_global.setValue(sizepre_global,u);
  sizepre_global=sizepre_global+1;

  color_global.setValue(u, gray_global);

  time_stamp_global = time_stamp_global + 1;

  d_global.setValue(u,time_stamp_global);


  iVector tempns(NNODES); // maxnodes
  iVector ns(NNODES);
  int NumberOfNeighboors;
  int vist=0;
  int index=0;
  int j;
  int v;
  int neighboorX;
  tempns.set(-1);
  ns.set(-1);
  NumberOfNeighboors=neighboor(u,tempns,Ad);


  for(i=0;i<NumberOfNeighboors;i++){
	  neighboorX=tempns.getValue(i);
      vist=0;
      for(j=0;j<NNODES;j++){
		  if(pred_global.getValue(j)==neighboorX)
	         vist=1;
      }
     if(vist==0){
		 ns.setValue(index,neighboorX);
         index=index+1;
      }
    }

  NumberOfNeighboors=index;


  for(i=0;i<NumberOfNeighboors;i++){
	  v=ns.getValue(i);
	if(color_global.getValue(v)== white_global) {
		pred_global.setValue(v,u);
        Dfs_Visit(v,Ad);
	 }

	else if(color_global.getValue(v)==gray_global){
         cycle_global=1;
    }
	else if(color_global.getValue(v)==black_global)
       break;
  }

  color_global.setValue(u,black_global);
  post_global.setValue(sizepost_global,u);
  sizepost_global=sizepost_global+1;
  time_stamp_global = time_stamp_global + 1;
  f_global.setValue(u,time_stamp_global);

}

// Need to make sure works wit arbritary trees
int InferenceEngineBP::findNbrs(int nodeI, iVector& NBRS)
{
  int i;
  int index=0;
  int NumNeighboors=0;
  int value;
  NBRS.create(NNODES, COLVECTOR, -1);
  int index2=1;
  for(i=nodeI+1;i<NNODES;i++)
    {
		value=AdjacencyMatrix.getValue(nodeI,i);
		index2=1;
		if(value>0){
			NBRS.setValue(index,index2+nodeI);
            index=index+1;
			index2=index2+1;
            NumNeighboors=NumNeighboors+1;
      }
    }
  return NumNeighboors;
}


// Utility Functions; most of them are just call dMatrix and vector class

void InferenceEngineBP::Transpose(dMatrix& M, dMatrix& MTranspose)
{
	int i;
    int j;
	for(i=0;i<M.getHeight();i++){
       for(j=0;j<M.getWidth();j++){
            MTranspose.setValue(j,i,M.getValue(i,j));
          }
    }
}

void InferenceEngineBP::LogNormalise(dVector& OldV, dVector& NewV)
{
  int i;
  int col;
  double sumold;
  double m1;
  double m2;
  double sub;
  double value;
//  int sizeV;
  NewV.create(NSTATES, COLVECTOR);
  for(i=0;i<NSTATES;i++){
	  NewV.setValue(i,OldV.getValue(i));
  }

  sumold=OldV.getValue(0);
  for(col=1;col<NSTATES;col++){
	  if(sumold >= OldV.getValue(col)){
		  m1=sumold;
		  m2=OldV.getValue(col);
	  } else {
		  m1=OldV.getValue(col);
          m2=sumold;
	  }
	  sub=m2-m1;
	  sumold=m1 + log(1 + exp(sub));
  }
  for(i=0;i<NSTATES;i++){
	  value=NewV.getValue(i)-sumold;
	  NewV.setValue(i,value);
  }
}

void InferenceEngineBP::LogNormalise2(dMatrix& OldM, dMatrix& NewM)
{
  int i;
  int j;
//  int k;
  int col;
  double sumold;
  double m1;
  double m2;
  double sub;
  double value;
//  int sizeV;
  dVector Temp;
  dVector NewV;
  int index1;
  index1=0;
  NewM.create(OldM.getWidth(),OldM.getHeight());
  Temp.create(OldM.getHeight()*OldM.getWidth());
  for(i=0;i<OldM.getHeight();i++){
	  for(j=0;j<OldM.getWidth();j++){
		  Temp.setValue(index1,OldM.getValue(i,j));
		  index1=index1+1;
	  }
  }
  NewV = Temp;
  sumold=Temp.getValue(0);
  for(col=1;col<index1;col++){
	  if(sumold >= Temp.getValue(col)){
		  m1=sumold;
		  m2=Temp.getValue(col);
	  } else {
		  m1=Temp.getValue(col);
          m2=sumold;
	  }
	  if(!wxFinite(m1) && !wxFinite(m2))
		  sub = 0;
	  else
		  sub=m2-m1;
	  sumold=m1 + log(1 + exp(sub));
   }
  for(i=0;i<index1;i++){
	  value=NewV.getValue(i)-sumold;
	  NewV.setValue(i,value);
  }

  index1=0;
  for(i=0;i<OldM.getHeight();i++){
	  for(j=0;j<OldM.getWidth();j++){
		  value=NewV.getValue(index1);
		  NewM.setValue(i,j,value);
		  index1=index1+1;
	  }
  }
}


