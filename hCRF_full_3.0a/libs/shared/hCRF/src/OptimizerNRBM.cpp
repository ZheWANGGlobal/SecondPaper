//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Non-convex Regularized Bundle Method (NRBM)[1], based on [2].
//
// Brief description :
//
//     min 0.5*lambda*((w-wreg).*reg)*((w-wreg).*reg)' + R(w)  (a)
//
// We introduce the new variable 
//     wnew = (w-wreg).* reg  <=> w = wnew ./ reg + wreg
// then R(w) = R( wnew ./ reg + wreg) = Rn(wnew)
//
// Minimizing (a) is equivalent to minimizing
//
//     min 0.5*lambda*wnew*wnew' + Rn(wnew)                    (b)
//
// where Rn(wnew) = R( wnew ./ reg + wreg) 
//
// The gradient is computed by: 
// d Rn(wnew) / d wnew 
//  = d R( wnew ./ reg + w0) / d wnew 
//  = (d R(w) / d w) * ( d w / d wnew ) 
//  = (d R(w) / d wnew) ./ reg
//
// [1] Do and Artieres, "Large Margin Training for Hidden Markov
//     Models with Partially Observed States." ICML 2009.
// [2] http://www.idiap.ch/~do/pmwiki/pmwiki.php/Main/Codes
//
//
// Yale Song (yalesong@csail.mit.edu)
// July, 2011

#include "optimizer.h"  

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

///////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR / DESTRUCTOR
//
OptimizerNRBM::OptimizerNRBM()
: Optimizer()
, m_Model(NULL)
, m_Dataset(NULL)
, m_Evaluator(NULL)
, m_Gradient(NULL)
{
}

OptimizerNRBM::~OptimizerNRBM()
{}


///////////////////////////////////////////////////////////////////////////
// PUBLIC
//

void OptimizerNRBM::optimize(Model *m, DataSet *X, Evaluator *eval, Gradient *grad)
{ 
	m_Model = m;
	m_Dataset = X;
	m_Evaluator = eval;
	m_Gradient= grad;
	
	int nbLabels = X->searchNumberOfSequenceLabels();
	int nbWeights = m_Model->getWeights()->getLength();
		
	dVector w0 = *(m->getWeights()); // initial solution
	dVector wstar(nbWeights);		 // optimal solution 
	
	reg_params p_reg;
	p_reg.lambda = m->getRegL2Sigma(); //1e-3;
	if( p_reg.lambda==0.f ) p_reg.lambda = 1e-3;
	p_reg.wreg.resize(1,nbWeights);	
	p_reg.reg.resize(1,nbWeights); p_reg.reg.set(1.0);
	
	nrbm_hyper_params p_nrbm;
	p_nrbm.bComputeGapQP	= false;
	p_nrbm.bCPMinApprox		= false;
	p_nrbm.bLineSearch		= true;
	p_nrbm.bRconvex			= false;
	p_nrbm.bRpositive		= true;
	p_nrbm.epsilon			= 1e-2;
	p_nrbm.maxNbCP			= 200;
	p_nrbm.maxNbIter		= maxit; // member variable
	
	// Start the NRBM optimization
	NRBM(w0, wstar, p_reg, p_nrbm);	

	// Save the optimal weights
	m->setWeights(wstar);	
}



///////////////////////////////////////////////////////////////////////////
// PRIVATE
//
// 

// R(w) = R( wnew ./ reg + wreg) = Rn(wnew)
void OptimizerNRBM::NRBM(dVector w0, dVector& wbest, 
						 const reg_params p_reg, const nrbm_hyper_params p_nrbm)
{
	dVector wn0(w0); wn0.subtract(p_reg.wreg); wn0.eltMpy(p_reg.reg); // wn0=(w0-wreg).*reg;	
	dVector wnbest(w0.getLength());	// return value from NRBM_kernel

	// Run NRBM
	NRBM_kernel(wn0, wnbest, p_reg, p_nrbm);

	// compute (wbest = wnbest ./ reg + wreg);
	wbest.set(wnbest);
	wbest.eltDiv(p_reg.reg);
	wbest.add(p_reg.wreg);	
}


// Basic non-convex regularized bundle method for solving the unconstrained problem
// min_w 0.5 * lambda * ||w||^2 + R(w)
void OptimizerNRBM::NRBM_kernel(dVector w0, dVector& wbest, 
								const reg_params p_reg, const nrbm_hyper_params p_nrbm)
{
	int nbWeights = w0.getLength();
	int i,r,c, tbest, cpbest, nbNew, numfeval;
	double s, sum_dist, fcurrent, fbest, Rbest, dual, fstart, astar, gap;
	
	dMatrix gVec, Q, Qtmp, newW, newGrad;
	dVector w,alpha,bias,cum_dist,Wtmp, newGrad0, newF;
	uVector cp_iter, inactive;
	dVector wLineSearch(nbWeights), gLineSearch(nbWeights), w1(nbWeights), g1(nbWeights); // for linesearch

	// [1] Variable initialization
	gVec.resize(p_nrbm.maxNbCP,nbWeights);		// set of gradients
	Q.resize(p_nrbm.maxNbCP,p_nrbm.maxNbCP);	// precompute gVec*gVec'

	w.set(w0);							// Current solution 
	alpha.resize(1,p_nrbm.maxNbCP);		// lagrangian multipliers
	bias.resize(1,p_nrbm.maxNbCP);		// bias term
	cum_dist.resize(1,p_nrbm.maxNbCP);	// cumulate distance
	cp_iter.resize(1,p_nrbm.maxNbCP);	// the iteration where cp is built
	inactive.resize(1,p_nrbm.maxNbCP);	// number of consecutive iteration that the cutting plane is inactive (alpha=0)		
	Wtmp.resize(1,nbWeights);			// should always be the same size (for efficiency)
	
	inactive.set(p_nrbm.maxNbCP); 	
	cp_iter.set(1);	
	cp_iter[p_nrbm.maxNbCP-1] = p_nrbm.maxNbIter+1; // the last slot is reserved for aggregation cutting plane

	s = 0;
	sum_dist = 0;
	fbest = DBL_MAX;
	Rbest = DBL_MAX;
	dual  = -DBL_MAX; 

	tbest  = 0;	// t, iteration index
	cpbest = 0;	// cp, cutting plane index
	
	// [2] Workspace for cutting plane to be added
	wolfe_params p_wolfe;
	p_wolfe.maxNbIter = 5;
	p_wolfe.a0 = 0.01;	
	p_wolfe.a1 = 0.5;
	p_wolfe.c1 = 1.000e-04;
	p_wolfe.c2 = 0.9;
	p_wolfe.amax = 1.1;
	
	newW.resize(p_wolfe.maxNbIter+1,nbWeights);
	newF.resize(1,p_wolfe.maxNbIter+1);
	newGrad.resize(p_wolfe.maxNbIter+1,nbWeights);
	newGrad0.resize(1,nbWeights);

	FGgrad_objective(w0, fstart, newGrad0, p_reg);	
	numfeval = 1; // counts num of inference performed
	
	newF[0] = fstart;
	for(int c=0; c<nbWeights; c++) newGrad(c,0) = newGrad0[c];
	for(int c=0; c<nbWeights; c++) newW(c,0) = w0[c];
	
	astar = p_wolfe.a0;
	gap = DBL_MAX; 
	nbNew = 1; // one cutting plane to be added in the next iteration
	
	// [3] Main loop	 
	std::list<int>::iterator it, it_a, it_b;
	std::list<std::pair<int,int> >::iterator it_pair;
	int t;
	for(t=0; t<p_nrbm.maxNbIter; t++) 
	{
		// ---------------------------------------------------------------------------------		
		// [3-1] Find memory slots of new cutting planes
		// ---------------------------------------------------------------------------------
		std::list<std::pair<int,int> > inactive_cp;
		for(i=0; i<inactive.getLength(); i++) 
			inactive_cp.push_back( std::pair<int,int>(i,inactive[i]*p_nrbm.maxNbIter*10-cp_iter[i]) );		
		inactive_cp.sort(Desc); 
		
		std::list<int> listCP, listCPold, listCPnew;
		for(i=0; i<inactive.getLength(); i++)
			if( inactive[i]<p_nrbm.maxNbCP ) 
				listCPold.push_back(i);
		for(i=0, it_pair=inactive_cp.begin(); i<nbNew; i++, it_pair++)
			listCPnew.push_back( (*it_pair).first );

		listCPold = matlab_setdiff(listCPold,listCPnew);
		for(it=listCPnew.begin(); it!=listCPnew.end(); it++) {
			inactive[*it] = 0; 
			cp_iter[*it] = t;
		}
		for(i=0; i<inactive.getLength(); i++)
			if( inactive[i]<p_nrbm.maxNbCP ) 
				listCP.push_back(i);
		
		// ---------------------------------------------------------------------------------
		// [3-2] Precompute Q for new cutting planes
		// ---------------------------------------------------------------------------------		
		// gVec(:,listCPnew) = newGrad(:,1:nbNew) - lambda*newW(:,1:nbNew);
		for(i=0, it=listCPnew.begin(); i<nbNew; i++, it++) 
			for(r=0; r<nbWeights; r++) 
				 gVec(r,*it) = newGrad(r,i) - p_reg.lambda*newW(r,i);   
		
		// Q(listCPnew,:) = gVec(:,listCPnew)' * gVec ;
		for(it=listCPnew.begin(); it!=listCPnew.end(); it++) {
			for(c=0; c<p_nrbm.maxNbCP; c++) {
				Q(*it,c) = 0;
				for(r=0; r<nbWeights; r++) {
					// gVec is very sparse, don't multiply unless non-zero
					if( gVec(r,c)==0 ) continue; 
					Q(*it,c) += gVec(r,*it) * gVec(r,c);					
				}
			}
		}			
		//Q(:,listCPnew) = Q(listCPnew,:)';
		Qtmp.resize((int)listCPnew.size(),p_nrbm.maxNbCP);
		for(i=0,it=listCPnew.begin(); it!=listCPnew.end(); i++,it++)
			for(c=0; c<p_nrbm.maxNbCP; c++)
				Qtmp(c,i) = Q(*it,c);
		for(i=0,it=listCPnew.begin(); it!=listCPnew.end(); i++,it++)
			for(r=0; r<p_nrbm.maxNbCP; r++)
				Q(r,*it) = Qtmp(r,i);

		// ---------------------------------------------------------------------------------
		// [3-3] Precompute Q for aggregation cutting plane.
		//       This part could be optimized by working only on Q
		// ---------------------------------------------------------------------------------
		// Q(maxCP,:) = gVec(:,maxCP)' * gVec;		
		for(c=0; c<p_nrbm.maxNbCP; c++) {
			Q(p_nrbm.maxNbCP-1,c) = 0;
			for(r=0; r<nbWeights; r++)
				Q(p_nrbm.maxNbCP-1,c) += gVec(r,p_nrbm.maxNbCP-1) * gVec(r,c);
		}
		// Q(:,maxCP) = Q(maxCP,:)';
		Qtmp.resize(1,p_nrbm.maxNbCP);
		for(c=0; c<p_nrbm.maxNbCP; c++)
			Qtmp(c,0) = Q(p_nrbm.maxNbCP-1,c);
		for(r=0; r<p_nrbm.maxNbCP; r++)
			Q(r,p_nrbm.maxNbCP-1) = Qtmp(r,0);


		// ---------------------------------------------------------------------------------
		// [3-4] Add each cutting plane to bundle
		// ---------------------------------------------------------------------------------
		fcurrent = 0;
		dVector wbestold(wbest);
		for(int k=0; k<nbNew; k++) {
			double reg_val, bias_val, Remp, dist, score, gamma, U, L;
			for(i=0; i<nbWeights; i++)
				Wtmp[i] = newW(i,k);
			reg_val = (0.5*p_reg.lambda) * Wtmp.l2Norm(false);
			Remp = newF[k] - reg_val;
			it=listCPnew.begin(); std::advance(it,k); int idx_cp=*it; // idx_cp = j (in original lib)
			bias_val=0;
			for(i=0; i<nbWeights; i++)
				bias_val += gVec(i,idx_cp)*newW(i,k);
			bias[idx_cp] = Remp - bias_val;
			fcurrent = newF[k];
			//printf("t=%d, k=%d, idx_cp=%d, fcurrent=%.3f, reg=%.3f, Remp=%.3f, gap=%f%\n", t,k,idx_cp,fcurrent,reg_val,Remp,gap*100/fabs(fbest));
			for(i=0; i<nbWeights; i++)
				Wtmp[i] = wbest[i]-newW(i,k);
			cum_dist[idx_cp] = Wtmp.l2Norm(false);

			if( fbest > fcurrent ) {
				fbest = fcurrent;
				Rbest = Remp;
				for(i=0; i<nbWeights; i++)
					wbest[i] = newW(i,k);
				tbest = t;
				cpbest = idx_cp;
				for(i=0; i<nbWeights; i++)
					Wtmp[i] = wbest[i]-wbestold[i];
				dist = Wtmp.l2Norm(false);
				sum_dist = sum_dist + dist;
				//printf("  norm_best=%f, dist=%f, sum_dist=%f\n", wbest.l2Norm(), dist, sum_dist);
				cum_dist[idx_cp] = 0;
			}

			// For non-convex optimization, solve conflict
			if( !p_nrbm.bRconvex ) {
				if( cpbest == idx_cp ) { // CASE 1: DESCENT STEP
					// list = [listCPold;listCPnew(1:k-1);
					std::list<int> listCPo, listCPn;
					for(it=listCPold.begin(); it!=listCPold.end(); it++)
						listCPo.push_back(*it);
					for(i=0, it=listCPnew.begin(); i<k-1 && it!=listCPnew.end(); it++)
						listCPn.push_back(*it);
					
					for(i=0, it=listCPo.begin(); it!=listCPo.end(); it++) {
						score = (0.5*p_reg.lambda)*wbest.l2Norm(false) + bias[i];
						for(int j=0; j<nbWeights; j++)
							score += gVec(j,i)*wbest[j];
						gamma = MAX(0,score - fbest + (0.5*p_reg.lambda)*cum_dist[i]);
						bias[i] -= gamma;					
					}
					for(i=0, it=listCPn.begin(); it!=listCPn.end(); it++) {
						score = (0.5*p_reg.lambda)*wbest.l2Norm(false) + bias[i];
						for(int j=0; j<nbWeights; j++)
							score += gVec(j,i)*wbest[j];
						gamma = MAX(0,score - fbest + (0.5*p_reg.lambda)*cum_dist[i]);
						bias[i] -= gamma;					
					}
				}
				else { // CASE 2: NULL STEP
					// Estimate g_t at w_tbest
					dist = cum_dist[idx_cp];
					score = (0.5*p_reg.lambda)*dist + bias[idx_cp];
					for(i=0; i<nbWeights; i++)
						score += gVec(i,idx_cp)*wbest[i];
					if( score > Rbest ) { // CONFLICT!
						// Solve conflict by descent g_t so that g_t(w_t) = fbest
						U = Rbest - (0.5*p_reg.lambda)*dist;
						L = fbest - reg_val;
						for(int j=0; j<nbWeights; j++) {
							U -= gVec(j,idx_cp) * wbest[j];
							L -= gVec(j,idx_cp) * newW(j,k);
						}
						//printf("NULL_STEP_CONFLICT Rbest=%f, score=%f, L=%f, U=%f, dist=%f\n", Rbest, score, L, U, dist);
						if( L<=U ) {
							//printf("NULL_STEP_CONFLICT LEVEL_1\n");
							bias[idx_cp] = L;
						}
						else {
							//printf("NULL_STEP_CONFLICT LEVEL_2\n");
							// gVec(:,j) = - lambda * wbest;
							for(i=0; i<nbWeights; i++)
								gVec(i,idx_cp) = -p_reg.lambda*wbest[i];
							// bias(j) = fbest - reg - gVec(:,j)'*newW(:,k);
							bias[idx_cp] = fbest - reg_val;
							for(i=0; i<nbWeights; i++)
								bias[idx_cp] -= gVec(i,idx_cp)*newW(i,k);							
							// Q(j,:) = gVec(:,j)' * gVec;
							for(i=0; i<p_nrbm.maxNbCP; i++) {
								Q(idx_cp,i) = 0;
								for(int j=0; j<nbWeights; j++)
									Q(idx_cp,i) += gVec(j,idx_cp) * gVec(j,i);
							}
							// Q(:,j) = Q(j,:)';
							Qtmp.resize(1,p_nrbm.maxNbCP);
							for(i=0; i<p_nrbm.maxNbCP; i++)
								Qtmp(i,0) = Q(idx_cp,i);
							for(i=0; i<p_nrbm.maxNbCP; i++)
								Q(i,idx_cp) = Qtmp(i,0);
						}
						score = (0.5*p_reg.lambda)*dist + bias[idx_cp];
						for(i=0; i<nbWeights; i++)
							score += gVec(i,idx_cp)*wbest[i];
						//printf("new_score = %f\n", score);
					}
				}
			}
		}

		// ---------------------------------------------------------------------------------
		// [3-5] Solve QP program
		// ---------------------------------------------------------------------------------
		// [alpha(listCP),dual] = minimize_QP(lambda,Q(listCP,listCP),bias(listCP),Rpositive ,epsilon);
		// Qtmp = Q(listCP,listCP);
		int size_tmp = (int)listCP.size();
		Qtmp.resize(size_tmp,size_tmp);
		for(r=0, it_a=listCP.begin(); it_a!=listCP.end(); r++, it_a++)
			for(c=0, it_b=listCP.begin(); it_b!=listCP.end(); c++, it_b++)
				Qtmp(r,c) = Q(*it_a,*it_b);
		
		dVector V_tmp(size_tmp), bias_tmp(size_tmp);
		for(i=0, it=listCP.begin(); it!=listCP.end(); i++, it++) {
			V_tmp[i] = alpha[*it];
			bias_tmp[i] = bias[*it];
		} 
		
		minimize_QP(p_reg.lambda, Qtmp, bias_tmp, p_nrbm.bRpositive, p_nrbm.epsilon, V_tmp, dual);
		for(i=0, it=listCP.begin(); it!=listCP.end(); i++, it++) 
			alpha[*it] = V_tmp[i]; 

		// ---------------------------------------------------------------------------------
		// [3-6] Get QP program solution, update inactive countings and the weight w
		// ---------------------------------------------------------------------------------	
		std::list<int> listA, listI, listCPA;
		for(i=0, it=listCP.begin(); it!=listCP.end(); i++, it++) { 
			if( V_tmp[i]>0 ) { 
				inactive[*it]=0;
				listA.push_back(i); 
				listCPA.push_back(*it); 
			}
			else if( V_tmp[i]==0 ) { 
				inactive[*it]++;
				listI.push_back(i); 
			} 
		}
		if( listCPA.size()>0 ) { // CHECK IF THIS IS OKAY!!!!! 
			V_tmp.resize(1,(int)listCPA.size());
			for(i=0,it=listCPA.begin(); it!=listCPA.end(); i++,it++)
				V_tmp[i] = -alpha[*it]/p_reg.lambda; // warning: V_tmp is overwritten here
			w_sum_row(gVec, V_tmp, listCPA, w);	
		}
		inactive[cpbest]=0; // make sure that the best point is always in the set
		

		// ---------------------------------------------------------------------------------
		// [3-7] Gradient aggregation
		// ---------------------------------------------------------------------------------
		for(i=0; i<nbWeights; i++)
			gVec(i,p_nrbm.maxNbCP-1) = -p_reg.lambda * w[i];
		bias[p_nrbm.maxNbCP-1] = dual + 0.5*p_reg.lambda*w.l2Norm(false);
		V_tmp.resize(1,(int)listCP.size());
		for(i=0,it=listCP.begin(); it!=listCP.end(); i++,it++)
			V_tmp[i] = cum_dist[*it]; // warning: V_tmp is overwritten here
		for(i=0,it=listCP.begin(); it!=listCP.end(); i++,it++)
			cum_dist[p_nrbm.maxNbCP-1] = alpha[*it] * V_tmp[i];		
		inactive[p_nrbm.maxNbCP-1]=0; // make sure that aggregation cp is always active 
		

		// ---------------------------------------------------------------------------------
		// [3-8] Estimate the gap of approximated dual problem
		// ---------------------------------------------------------------------------------
		if( p_nrbm.bComputeGapQP ) {
			// scoreQP = (w' * gVec)' + bias;
			dVector scoreQP(bias);
			for(i=0; i<p_nrbm.maxNbCP; i++) 
				for(int j=0; j<nbWeights; j++)
					scoreQP[i] += w[j] * gVec(j,i);
			double primalQP = 0.5*p_reg.lambda*w.l2Norm(false);
			double max_scoreQP = -DBL_MAX;
			for(it=listCP.begin(); it!=listCP.end(); it++)
				if( scoreQP[*it] > max_scoreQP ) max_scoreQP = scoreQP[*it];
			if( !(max_scoreQP<0&&p_nrbm.bRpositive) ) primalQP += max_scoreQP;
			double gapQP = primalQP - dual;
 			//printf("  quadratic_programming: primal=%f, dual=%f, gap=%f\n",primalQP,dual,gapQP*100/fabs(primalQP));
		} 
		gap = fbest-dual;
		// skip the freport thingy
		
		// ---------------------------------------------------------------------------------
		// [3-9] Output
		// ---------------------------------------------------------------------------------
		if( m_Model->getDebugLevel() >= 1 ) {
			printf("  t=%d, nfeval=%d, f=%f, f*=%f, R*=%f, gap=%.2f%\n", t, numfeval,
									fcurrent, fbest, Rbest, gap*100/fabs(fbest));
		}
		if( gap/fabs(fbest)<p_nrbm.epsilon || gap<1e-6 || t>=p_nrbm.maxNbIter )
			break;
		if( !p_nrbm.bLineSearch ) {
			nbNew = 1;
			double newF0 = 0;
			FGgrad_objective(w,newF0,newGrad0,p_reg);
			newF[0] = newF0;
			for(i=0; i<nbWeights; i++) 
				newGrad(i,0) = newGrad0[i];
			for(i=0; i<nbWeights; i++)
				newW(i,0) = w[i];
			numfeval++;
			continue;
		}
		
		// ---------------------------------------------------------------------------------
		// [3-10] Perform line search from wbest to w
		// ---------------------------------------------------------------------------------
		dVector search_direction(w); search_direction.subtract(wbest);
		double norm_direction = search_direction.l2Norm();
		double astart;
		if( p_nrbm.bCPMinApprox || t==0 ) {
			astart = 1.0;
		}
		else {
			astart = MIN(astar/norm_direction,1.0);
			if( astart==0 ) astart=1.0;
		}
		dVector g0(wbest); g0.multiply(p_reg.lambda); 
		for(i=0; i<nbWeights; i++) 
			g0[i] += gVec(i,cpbest);
		double fLineSearch,f1;
		numfeval += myLineSearchWolfe(
						wbest,fbest,g0,search_direction,astart,p_reg,p_wolfe,
						astar,wLineSearch,fLineSearch,gLineSearch,w1,f1,g1); 
		if( f1!=fLineSearch ) {
			nbNew = 2;
			newF[0]=f1; newF[1]=fLineSearch;
			for(i=0; i<nbWeights; i++) {
				   newW(i,0)=w1[i];	   newW(i,1)=wLineSearch[i];
				newGrad(i,0)=g1[i];	newGrad(i,1)=gLineSearch[i]; 
			}
		}
		else {
			nbNew = 1;
			newF[0] = fLineSearch;
			for(i=0; i<nbWeights; i++) {
				newW(i,0)=wLineSearch[i];
				newGrad(i,0)=gLineSearch[i];
			}
		}
		if( fbest<=fLineSearch && astart!=1 ) {
			numfeval++; nbNew++;
			double newF0=0; newGrad0.set(0); FGgrad_objective(w,newF0,newGrad0,p_reg);
			newF[nbNew-1] = newF0; 
			for(i=0; i<nbWeights; i++) {
				newGrad(i,nbNew-1) = newGrad0[i];
				newW(i,nbNew-1) = w[i];
			}
		} 
		astar = astar*norm_direction; // true step length
		//printf("> step_length = %f\n", astar);		 
	}
	lastFunctionError = fcurrent;
	lastNbIterations = t;
	printf("DONE_DRBM numfeval=%d\n", numfeval);
}

void OptimizerNRBM::FGgrad_objective(dVector w, double& fval, dVector& grad, const reg_params p_reg)
{
	// Evaluate
	double Rval = 0.0;
	fnew(w,Rval,grad,p_reg);
	
	// Set regularization values
	fval = (0.5*p_reg.lambda)*w.l2Norm(false) + Rval; 
	dVector reg(w); reg.multiply(p_reg.lambda); grad.add(reg);
}

void OptimizerNRBM::fnew(dVector wnew, double& Remp, dVector& gradwn, const reg_params p_reg)
{	
	dVector w(wnew); w.eltDiv(p_reg.reg); w.add(p_reg.wreg); // w = wnew./reg + wreg;
	Fgrad(w,Remp,gradwn,p_reg);
	gradwn.eltDiv(p_reg.reg);
}

void OptimizerNRBM::Fgrad(dVector w, double& F, dVector& grad, const reg_params p_reg)
{ 
	// Compute F and Grad from inference_engine and gradient_engine 
	m_Model->setWeights(w); 
	F = m_Gradient->computeGradient(grad,m_Model,m_Dataset,m_Model->isMaxMargin()); 

	// Make F value independent to the dataset size
	int nbSeq = (int)m_Dataset->size();
	F /= (double)nbSeq;
	grad.multiply(1/(double)nbSeq); 
}

void OptimizerNRBM::minimize_QP(double lambda, dMatrix Q, dVector B, bool Rpositive, double EPS,
								dVector& alpha, double& dual)
{ 
	int i,r,c;
	int size_var = Q.getHeight();
	if( size_var+Rpositive==1 ) { 
		alpha[0] = 1;
		dual = -0.5 * Q(0,0)/lambda + B[0];
		return;	
	}
	if( Rpositive ) {
		dMatrix Qtmp(Q); 
		dVector Btmp(B);
		Q.resize(Q.getWidth()+1,Q.getHeight()+1);
		B.resize(1,B.getLength()+1);
		for(i=0; i<Btmp.getLength(); i++)
			B[i] = Btmp[i];
		for(r=0; r<Qtmp.getHeight(); r++)
			for(c=0; c<Qtmp.getWidth(); c++)
				Q(r,c) = Qtmp(r,c);
	}
	double scale = Q.absmax() / (1000*lambda);
	double tmp = (scale!=0) ? 1/scale : DBL_MAX; 
	Q.multiply(tmp);
	B.multiply(tmp);

	dMatrix Qtmp(Q); Qtmp.multiply(1/lambda); // This must be positive definite 
	dVector Btmp(B); Btmp.negate(); Btmp.transpose();

	std::list<std::string> list_solvers;
	std::list<std::string>::iterator it;
	list_solvers.push_back(std::string("imdm"));
	list_solvers.push_back(std::string("kowalczyk"));
	list_solvers.push_back(std::string("keerthi"));
	
	qp_params p_qp;
	p_qp.absTolerance = 0.0;
	p_qp.relTolerance = 1e-2*EPS;
	p_qp.threshLB = DBL_MAX;

	int exit_flag = -1;
	for(int k=6; k<=10; k++) {		
		for(it=list_solvers.begin(); it!=list_solvers.end(); it++) 
		{
			// Call the QP solver
			p_qp.solver = (*it).c_str();
			p_qp.maxNbIter = (int)pow((float)10,(float)k);
			exit_flag = QP_kernel(Qtmp, Btmp, alpha, dual, p_qp);
			
			// verify the solution of the approximated dual problem.
			double eps = 1e-4;
			if( alpha.min()<-eps || fabs(alpha.sum()-1)>eps )
				exit_flag = 0;
			if( exit_flag>0 ) 
				break;
		}
		if( exit_flag!=0 )
			break;
	}
	
#if _DEBUG
	if( exit_flag==0 ) 
		printf("Warning: minimize_QP() solving QP of approx. problem failed. Did not reach enough accuracy\n");
#endif

	B.multiply(scale);
	dual = -dual * scale;
	if( Rpositive ) {
		dVector V_tmp(alpha);
		alpha.resize(1,size_var);
		for(i=0; i<size_var; i++)
			alpha[i] = V_tmp[i];
	}
}



void OptimizerNRBM::w_sum_row(dMatrix H, dVector a, std::list<int> idx, dVector &w)
{
	int M = H.getHeight(); 
	int N = H.getWidth();
	int Da = a.getLength();
	int Di = (int) idx.size();
	int Dw = w.getLength();

#if _DEBUG
	if( Di!=Da || Da>N || M!=Dw ) {
		printf("w_sum_row() dimension mismatch!");
		printf(" H[%d x %d], a[%d], idx[%d], w[%d]\n", M,N,Da,Di,Dw);
		getchar();
		exit(-1);
	}
#endif
	int i,j;
	std::list<int>::iterator it;
	for(j=0; j<M; j++) {
		w[j] = 0.0;
		for(i=0,it=idx.begin(); i<Da; i++,it++)
			w[j] += H(j,*it) * a[i];
	}
}

void OptimizerNRBM::w_sum_col(dMatrix H, dVector a, std::list<int> idx, dVector &w)
{
	int M = H.getHeight(); 
	int N = H.getWidth();
	int Da = a.getLength();
	int Di = (int) idx.size();
	int Dw = w.getLength();

#if _DEBUG
	if( Di!=Da || Da>N || N!=Dw ) {
		printf("w_sum_col() dimension mismatch!");
		getchar();
		printf(" H[%d x %d], a[%d], idx[%d], w[%d]\n", M,N,Da,Di,Dw);
		exit(-1);
	}
#endif
	int i,j;
	std::list<int>::iterator it;
	for(j=0; j<N; j++) {
		w[j] = 0.0;
		for(i=0,it=idx.begin(); i<Da; i++,it++)
			w[j] += H(*it,j) * a[i];
	}
}


// Reimplementation of some Matlab functions
std::list<int> OptimizerNRBM::matlab_setdiff(
	std::list<int> list_a, std::list<int> list_b)
{
	std::list<int> diff_sorted;
	std::list<int>::iterator it_a, it_b, it_c;
	for(it_a=list_a.begin(); it_a!=list_a.end(); it_a++) 
		diff_sorted.push_back(*it_a);
	for(it_b=list_b.begin(); it_b!=list_b.end(); it_b++) {
		for(it_c=diff_sorted.begin(); it_c!=diff_sorted.end(); ) {
			if( *it_b==*it_c ) 
				diff_sorted.erase(it_c++);
			else 
				it_c++;
		}
	}
	diff_sorted.sort();
	diff_sorted.unique(); 
	return diff_sorted;
} 
