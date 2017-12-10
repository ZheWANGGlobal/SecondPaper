//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// General QP Solver, based on STPRtoolbox [1]
//
// The Generalized Minimal Norm Problem to solve
//    min_x 0.5*x'*H*x + f'*x  subject to: sum(x) = 1 and x >= 0.
//
// H is symetric positive-definite matrix. The GMNP is a special
// instance of the Quadratic Programming (QP) task. The GMNP
// is solved by one of the following algorithms:
//    mdm        Mitchell-Demyanov-Malozemov
//    imdm       Improved Mitchell-Demyanov-Malozemov (default).
//    iimdm      Improved (version 2) Mitchell-Demyanov-Malozemov.
//    kozinec    Kozinec algorithm.
//    keerthi    Derived from NPA algorithm by Keerthi et al.
//    kowalczyk  Based on Kowalczyk's maximal margin perceptron.
//
// The optimization halts if one of the following stopping conditions is satisfied:
//	tmax <= t                -> exit_flag = 0
//	tolabs >= UB-LB          -> exit_flag = 1
//	tolrel*abs(UB) >= UB-LB  -> exit_flag = 2
//	thlb < LB                -> exit_flag = 3
// where t is number of iterations, UB/LB are upper/lower bounds on the optimal solution.
//
//  For more info refer to V.Franc: Optimization Algorithms for Kernel 
//  Methods. Research report. CTU-CMP-2005-22. CTU FEL Prague. 2005.
//  ftp://cmp.felk.cvut.cz/pub/cmp/articles/franc/Franc-PhD.pdf .
//
// Input:
//	H [dim x dim] Symmetric positive definite matrix.
//	c [dim x 1] Vector.
//	solver [string] GMNP solver: options are 'mdm', 'imdm', 'iimdm','kowalczyk','keerthi','kozinec'.
//	tmax [1x1] Maximal number of iterations.
//	tolabs [1x1] Absolute tolerance stopping condition.
//	tolrel [1x1] Relative tolerance stopping condition.
//	thlb [1x1] Threshold on lower bound.  
//
// Output:
//  alpha [dim x 1] Solution vector.
//  exitflag [1x1] Indicates which stopping condition was used:
//    UB-LB <= tolabs           ->  exit_flag = 1   Abs. tolerance.
//    UB-LB <= UB*tolrel        ->  exit_flag = 2   Relative tolerance.
//    LB > th                   ->  exit_flag = 3   Threshold on LB.
//    t >= tmax                 ->  exit_flag = 0   Number of iterations.
//  t [1x1] Number of iterations.
//  access [1x1] Access to elements of the matrix H.
//  History [2x(t+1)] UB and LB with respect to number of iterations.
//
// [1] http://cmp.felk.cvut.cz/cmp/software/stprtool/
//  
// Yale Song (yalesong@csail.mit.edu)
// October, 2011

#include "optimizer.h"  

#define HISTORY_BUF 1000000

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define INDEX(ROW,COL,DIM) ((COL*DIM)+ROW)

int OptimizerNRBM::QP_kernel(dMatrix H, dVector c, dVector& alpha, double& dual, qp_params p_qp)
{
	int dim = H.getWidth();

#ifdef _DEBUG
	if( H.getHeight()!=H.getWidth() ) {
		fprintf(stderr, "QP_kernel(): H must be squared.\n"); getchar(); exit(-1);
	}
	if( H.getHeight()!=c.getLength() ) {
		fprintf(stderr, "QP_kernel(): H and c dimension mismatch.\n"); getchar(); exit(-1);
	} 
#endif
	
	dVector diag_H(dim); for(int i=0; i<dim; i++) diag_H[i] = H(i,i);

	int exitflag = 0;
	if( !strcmp(p_qp.solver,"mdm") ) 
		exitflag = QP_mdm(H, diag_H, c, alpha, dual, p_qp);
	else if( !strcmp(p_qp.solver,"imdm") )
		exitflag = QP_imdm(H, diag_H, c, alpha, dual, p_qp);
	else if( !strcmp(p_qp.solver,"iimdm") )
		exitflag = QP_iimdm(H, diag_H, c, alpha, dual, p_qp);
	else if( !strcmp(p_qp.solver,"keerthi") )
		exitflag = QP_keerthi(H, diag_H, c, alpha, dual, p_qp);
	else if( !strcmp(p_qp.solver,"kowalczyk") )
		exitflag = QP_kowalczyk(H, diag_H, c, alpha, dual, p_qp);
	else if( !strcmp(p_qp.solver,"kozinec") )
		exitflag = QP_kozinec(H, diag_H, c, alpha, dual, p_qp);
	else {
		fprintf(stderr, "Unknown QP solver (%s).\n", p_qp.solver); getchar(); exit(-1);
	} 

	return exitflag;
	
}

int OptimizerNRBM::QP_mdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp)
{ 
	int dim = H.getHeight();
	
	// Return values
	alpha.resize(1,dim);
	dual = 0;

	// Variables
	double lambda, LB,UB, aHa,ac, tmp,tmp1, Huu,Huv,Hvv, beta,min_beta,max_beta; 
	int u,v, new_u,new_v, i,t,exitflag; //,History_size;
	dVector Ha; //, History;

	// Initialization
	Ha.create(dim);
	//History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
	//History.create(History_size*2);

	tmp1 = DBL_MAX;
	for(i=0; i<dim; i++) {
		tmp = 0.5*diag_H[i] + c[i];
		if( tmp1 > tmp ) {
			tmp1 = tmp;
			v = i;
		}
	}

	min_beta = DBL_MAX;
	for(i=0; i<dim; i++) {
		alpha[i] = 0;
		Ha[i] = H(i,v);
		beta = Ha[i] + c[i];
		if( beta < min_beta ) {
			min_beta = beta;
			u = i;
		}
	}
	
	alpha[v] = 1;
	aHa = diag_H[v];
	ac = c[v];

	UB = 0.5*aHa + ac;
	LB = min_beta - 0.5*aHa;
	t = 0;
	//History[INDEX(0,0,2)] = LB;
	//History[INDEX(1,0,2)] = UB;

	//printf("QP_mdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);

	// Stopping conditions
	if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
	else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
	else if( LB > p_qp.threshLB )					exitflag = 3;
	else											exitflag = -1;
	
	// Main QP Optimization Loop
	while( exitflag==-1 )
	{
		// Adaptation rule and update
		Huu = diag_H[u];
		Hvv = diag_H[v];
		Huv = H(v,u);

		lambda = (Ha[v]-Ha[u]+c[v]-c[u]) / (alpha[v]*(Huu-2*Huv+Hvv));
		if( lambda<0 ) lambda=0; else if( lambda>1 ) lambda=1;
		
		aHa = aHa + 2*alpha[v]*lambda*(Ha[u]-Ha[v]) 
				  + lambda*lambda*alpha[v]*alpha[v]*(Huu-2*Huv+Hvv);
		
		ac = ac + lambda*alpha[v]*(c[u]-c[v]);

		tmp = alpha[v];
		alpha[u] += lambda*alpha[v];
		alpha[v] -= lambda*alpha[v];

		UB = 0.5*aHa + ac;
		
		min_beta = DBL_MAX;
		max_beta = -DBL_MAX;
		for(i=0; i<dim; i++) {
			Ha[i] += lambda*tmp*(H(i,u)-H(i,v));
			beta = Ha[i] + c[i];
			if( alpha[i]!=0 && max_beta<beta ) {
				new_v = i; 
				max_beta = beta;
			}

			if( beta<min_beta ) {
				new_u = i;
				min_beta = beta;
			}
		}

		LB = min_beta - 0.5*aHa;
		u = new_u;
		v = new_v;

		// Stopping conditions
		if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
		else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
		else if( LB > p_qp.threshLB )					exitflag = 3;
		else if( t >= p_qp.maxNbIter )					exitflag = 0;

		//printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);

		// Store selected values
//		if( t < History_size ) {
//			History[INDEX(0,t,2)] = LB;
//			History[INDEX(1,t,2)] = UB;
//		} 
//		else 
//			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n"); 

		t++;
	}

	// Print info about last iteration
	//printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB); 

	dual = UB;
	return exitflag;
}

int OptimizerNRBM::QP_imdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp)
{ 
	int dim = H.getHeight();
	
	// Return values
	alpha.resize(1,dim);
	dual = 0;

	// Variables
	double lambda, LB,UB, aHa,ac, tmp,tmp1, Huu,Huv,Hvv, beta,min_beta, max_improv,improv; 
	int u,v, new_u, i,t,exitflag; //,History_size;
	dVector Ha; //, History;

	// Initialization
	Ha.create(dim);
	//History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
	//History.create(History_size*2);

	tmp1 = DBL_MAX;
	for(i=0; i<dim; i++) {
		tmp = 0.5*diag_H[i] + c[i];
		if( tmp1 > tmp ) {
			tmp1 = tmp;
			v = i;
		}
	}

	min_beta = DBL_MAX;
	for(i=0; i<dim; i++) {
		alpha[i] = 0;
		Ha[i] = H(i,v);
		beta = Ha[i] + c[i];
		if( beta < min_beta ) {
			min_beta = beta;
			u = i;
		}
	}
	
	alpha[v] = 1;
	aHa = diag_H[v];
	ac = c[v];

	UB = 0.5*aHa + ac;
	LB = min_beta - 0.5*aHa;
	t = 0;
	//History[INDEX(0,0,2)] = LB;
	//History[INDEX(1,0,2)] = UB;

	//printf("QP_imdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);

	// Stopping conditions
	if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
	else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
	else if( LB > p_qp.threshLB )					exitflag = 3;
	else											exitflag = -1;
	
	// Main QP Optimization Loop
	int u0 = u;
	while( exitflag==-1 )
	{
		// Adaptation rule and update
		Huu = diag_H[u];
		Hvv = diag_H[v];
		Huv = H(v,u0);

		lambda = (Ha[v]-Ha[u]+c[v]-c[u]) / (alpha[v]*(Huu-2*Huv+Hvv));
		if( lambda<0 ) lambda=0; else if( lambda>1 ) lambda=1;
		
		aHa = aHa + 2*alpha[v]*lambda*(Ha[u]-Ha[v]) 
				  + lambda*lambda*alpha[v]*alpha[v]*(Huu-2*Huv+Hvv);
		
		ac = ac + lambda*alpha[v]*(c[u]-c[v]);

		tmp = alpha[v];
		alpha[u] += lambda*alpha[v];
		alpha[v] -= lambda*alpha[v];

		UB = 0.5*aHa + ac;
		
		min_beta = DBL_MAX; 
		for(i=0; i<dim; i++) {
			Ha[i] += lambda*tmp*(H(i,u0)-H(i,v));
			beta = Ha[i] + c[i]; 

			if( beta<min_beta ) {
				new_u = i;
				min_beta = beta;
			}
		}

		LB = min_beta - 0.5*aHa;
		u = new_u; 
		u0 = u; 

		// Search for the optimal v while u is fixed
		max_improv = -DBL_MAX;
		for(i=0; i<dim; i++) {
			if( alpha[i]!=0 ) {
				beta = Ha[i] + c[i];
				if( beta >= min_beta ) {
					tmp = diag_H[u] - 2*H(i,u0) + diag_H[i];
					if( tmp!=0 ) {
						improv = (0.5*(beta-min_beta)*(beta-min_beta))/tmp;
						if( improv>max_improv ) {
							max_improv = improv;
							v = i;
						}
					}
				}
			}
		}

		// Stopping conditions
		if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
		else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
		else if( LB > p_qp.threshLB )					exitflag = 3;
		else if( t >= p_qp.maxNbIter )					exitflag = 0;

		//printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);

		// Store selected values
//		if( t < History_size ) {
//			History[INDEX(0,t,2)] = LB;
//			History[INDEX(1,t,2)] = UB;
//		} 
//		else 
//			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n"); 

		t++;
	}

	// Print info about last iteration
	//printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB); 

	dual = UB;
	return exitflag;
	return 0;
}

int OptimizerNRBM::QP_iimdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp)
{
	int dim = H.getHeight();
	
	// Return values
	alpha.resize(1,dim);
	dual = 0;

	// Variables
	double lambda, LB,UB, aHa,ac, tmp,tmp1, Huu,Huv,Hvv, 
		beta,min_beta,max_beta, max_improv1,max_improv2,improv; 
	int u,v, new_u,new_v, i,t,exitflag; //,History_size;
	dVector Ha; //, History;

	// Initialization
	Ha.create(dim);
	//History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
	//History.create(History_size*2);

	tmp1 = DBL_MAX;
	for(i=0; i<dim; i++) {
		tmp = 0.5*diag_H[i] + c[i];
		if( tmp1 > tmp ) {
			tmp1 = tmp;
			v = i;
		}
	}

	min_beta = DBL_MAX;
	for(i=0; i<dim; i++) {
		alpha[i] = 0;
		Ha[i] = H(i,v);
		beta = Ha[i] + c[i];
		if( beta < min_beta ) {
			min_beta = beta;
			u = i;
		}
	}
	
	alpha[v] = 1;
	aHa = diag_H[v];
	ac = c[v];

	UB = 0.5*aHa + ac;
	LB = min_beta - 0.5*aHa;
	t = 0;
	//History[INDEX(0,0,2)] = LB;
	//History[INDEX(1,0,2)] = UB;

	//printf("QP_mdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);

	// Stopping conditions
	if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
	else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
	else if( LB > p_qp.threshLB )					exitflag = 3;
	else											exitflag = -1;
	
	// Main QP Optimization Loop
	int u0 = u; 
	int v0 = v;
	while( exitflag==-1 )
	{
		// Adaptation rule and update
		Huu = diag_H[u];
		Hvv = diag_H[v];
		Huv = H(v,u0);

		lambda = (Ha[v]-Ha[u]+c[v]-c[u]) / (alpha[v]*(Huu-2*Huv+Hvv));
		if( lambda<0 ) lambda=0; else if( lambda>1 ) lambda=1;
		
		aHa = aHa + 2*alpha[v]*lambda*(Ha[u]-Ha[v]) 
				  + lambda*lambda*alpha[v]*alpha[v]*(Huu-2*Huv+Hvv);
		
		ac = ac + lambda*alpha[v]*(c[u]-c[v]);

		tmp = alpha[v];
		alpha[u] += lambda*alpha[v];
		alpha[v] -= lambda*alpha[v];

		UB = 0.5*aHa + ac;
		
		min_beta = DBL_MAX;
		max_beta = -DBL_MAX;
		for(i=0; i<dim; i++) {
			Ha[i] += lambda*tmp*(H(i,u0)-H(i,v0));
			beta = Ha[i] + c[i]; 

			if( beta<min_beta ) {
				new_u = i;
				min_beta = beta;
			}

			if( alpha[i]!=0 && max_beta<beta ) {
				new_v = i;
				max_beta = beta;
			}
		}

		LB = min_beta - 0.5*aHa; 
		u0 = new_u;
		v0 = new_v;

		// Search for the optimal v while u is fixed
		max_improv1 = max_improv2 = -DBL_MAX;
		for(i=0; i<dim; i++) {
			beta = Ha[i] + c[i];
			if( alpha[i]!=0 && beta>min_beta ) {
				tmp = diag_H[new_u] - 2*H(i,u0) + diag_H[i];
				if( tmp!=0 ) {
					if( (beta-min_beta)/(alpha[i]*tmp) < 1 ) 
						improv = (0.5*(beta-min_beta)*(beta-min_beta))/tmp;
					else
						improv = alpha[i]*(beta-min_beta) - 0.5*alpha[i]*alpha[i]*tmp;

					if( improv>max_improv1 ) {
						max_improv1 = improv;
						v = i;
					}
				}
			}

			if( max_beta>beta ) {
				tmp = diag_H[new_v] - 2*H(i,v0) + diag_H[i];
				if( tmp!=0 ) {
					if( (max_beta-beta)/(alpha[new_v]*tmp) < 1 ) 
						improv = (0.5*(max_beta-beta)*(max_beta-beta))/tmp;
					else
						improv = alpha[new_v]*(max_beta-beta) - 0.5*alpha[new_v]*alpha[new_v]*tmp;

					if( improv>max_improv2 ) {
						max_improv2 = improv;
						u = i;
					}					
				}
			}
		}

		if( max_improv1 > max_improv2 ) {
			u = new_u;
			v0 = v;
		}
		else {
			v = new_v;
			u = u0;
		}

		// Stopping conditions
		if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
		else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
		else if( LB > p_qp.threshLB )					exitflag = 3;
		else if( t >= p_qp.maxNbIter )					exitflag = 0;

		//printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);

		// Store selected values
//		if( t < History_size ) {
//			History[INDEX(0,t,2)] = LB;
//			History[INDEX(1,t,2)] = UB;
//		} 
//		else 
//			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n"); 

		t++;
	}

	// Print info about last iteration
	//printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB); 

	dual = UB;
	return exitflag;
	return 0;
}

int OptimizerNRBM::QP_keerthi(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp)
{
	int dim = H.getHeight();
	
	// Return values
	alpha.resize(1,dim);
	dual = 0;

	// Variables
	double LB,UB, aHa,ac, tmp,tmp1, Huu,Huv,Hvv, beta,min_beta,max_beta; 
	double den,gamma,omega,a1,a2,a3,a4,a5,x10,x11,x12,x13,x20,x22,x23,x30,x33;
	double UB123,gamma1,gamma2,gamma3,tmp_aHa1,tmp_aHa2,tmp_aHa3,tmp_ac1,tmp_ac2,tmp_ac3,UB1,UB2,UB3;
	int nearest_segment, u,v, i,t,exitflag; //,History_size;
	dVector Ha; //, History;

	// Initialization
	Ha.create(dim);
	//History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
	//History.create(History_size*2);

	tmp1 = DBL_MAX;
	for(i=0; i<dim; i++) {
		tmp = 0.5*diag_H[i] + c[i];
		if( tmp1 > tmp ) {
			tmp1 = tmp;
			v = i;
		}
	}

	min_beta = DBL_MAX;
	for(i=0; i<dim; i++) {
		alpha[i] = 0;
		Ha[i] = H(i,v);
		beta = Ha[i] + c[i];
		if( beta < min_beta ) {
			min_beta = beta;
			u = i;
		}
	}
	
	alpha[v] = 1;
	aHa = diag_H[v];
	ac = c[v];

	UB = 0.5*aHa + ac;
	LB = min_beta - 0.5*aHa;
	t = 0;
	//History[INDEX(0,0,2)] = LB;
	//History[INDEX(1,0,2)] = UB;

	//printf("QP_mdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);

	// Stopping conditions
	if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
	else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
	else if( LB > p_qp.threshLB )					exitflag = 3;
	else											exitflag = -1;
	
	// Main QP Optimization Loop
	while( exitflag==-1 )
	{
		// Adaptation rule and update
		Huu = diag_H[u];
		Hvv = diag_H[v];
		Huv = H(v,u);

		x11 = aHa;
		x12 = Ha[u];
		x13 = aHa + alpha[v]*(Ha[u]-Ha[v]);
		x22 = Huu;
		x23 = Ha[u] + alpha[v]*(Huu-Huv);
		x33 = aHa + 2*alpha[v]*(Ha[u]-Ha[v]) + alpha[v]*alpha[v]*(Huu-2*Huv+Hvv);

		x10 = ac;
		x20 = c[u];
		x30 = ac + alpha[v]*(c[u]-c[v]);

		a1 = x11 - x12 - x13 + x23;
		a2 = x11 - 2*x12 + x22;
		a3 = x12 - x11 + x20 - x10;
		a4 = x11 - 2*x13 + x33;
		a5 = x13 - x11 + x30 - x10;		

		den = a1*a1 - a2*a4;
		if( den ) {
			gamma = (a3*a4 - a1*a5)/den;
			omega = (a2*a5 - a3*a1)/den;

			if( gamma>0 && omega>0 && 1-gamma-omega>0 ) {
				// Ha = Ha*(1-gamma) + H(:,u)*(gamma+alpha(v)*omega)-H(:,v)*alpha(v)*omega;
				tmp = alpha[v]*omega;
				for(i=0; i<dim; i++) 
					Ha[i] = Ha[i]*(1-gamma) + H(i,u)*(gamma+tmp) - H(i,v)*tmp;
				 
				// aHa = (1-omega-gamma)^2*x11 + gamma^2*x22 + omega^2*x33 + ...
				//       2*(1-omega-gamma)*gamma*x12 + 2*(1-omega-gamma)*omega*x13 + ...
				//       2*gamma*omega*x23; 
				aHa = (1-omega-gamma)*(1-omega-gamma)*x11 + gamma*gamma*x22
					 + omega*omega*x33 + 2*(1-omega-gamma)*gamma*x12 
					 + 2*(1-omega-gamma)*omega*x13 + 2*gamma*omega*x23;
				ac = (1-gamma-omega)*x10 + gamma*x20 + omega*x30;

				
  			    // alpha1 = zeros(dim,1);
 			    // alpha1(u) = 1;
			    // alpha2 = alpha;
			    // alpha2(u) = alpha(u)+alpha(v);
			    // alpha2(v) = 0;
			    // alpha = alpha*(1-gamma-omega) + alpha1*gamma + alpha2*omega;
				for(i=0; i<dim; i++) 
					alpha[i] *= (1-gamma);

				alpha[u] += (gamma + tmp);
				alpha[v] -= tmp;

				UB123 = 0.5*aHa + ac;
			}
			else {
				UB123 = DBL_MAX;
			}
		}
		else {
			UB123 = DBL_MAX;
		}

		if( UB123 == DBL_MAX ) {
			// line segment between alpha and alpha1 
			gamma1   = (x11-x12+x10-x20)/(x11-2*x12+x22);
			gamma1   = MIN(1,gamma1);
			tmp_aHa1 = (1-gamma1)*(1-gamma1)*x11 + 2*gamma1*(1-gamma1)*x12 + gamma1*gamma1*x22;
			tmp_ac1  = (1-gamma1)*x10 + gamma1*x20;
			UB1      = 0.5*tmp_aHa1 + tmp_ac1;

			// line segment between alpha and alpha2 
			gamma2   = (x11-x13+x10-x30)/(x11-2*x13+x33);
			gamma2   = MIN(1,gamma2);
			tmp_aHa2 = (1-gamma2)*(1-gamma2)*x11 + 2*gamma2*(1-gamma2)*x13 + gamma2*gamma2*x33;
			tmp_ac2  = (1-gamma2)*x10 + gamma2*x30;
			UB2      = 0.5*tmp_aHa2 + tmp_ac2;
			
			//  line segment between alpha1 and alpha2
			den = (x22 - 2*x23 + x33);
			if( den ) {
				gamma3 = (x22-x23+x20-x30)/den;
				if( gamma3 > 1 ) gamma3 = 1;
				if( gamma3 < 0 ) gamma3 = 0;
				tmp_aHa3 = (1-gamma3)*(1-gamma3)*x22 + 2*gamma3*(1-gamma3)*x23 + gamma3*gamma3*x33;
				tmp_ac3  = (1-gamma3)*x20 + gamma3*x30;
				UB3      = 0.5*tmp_aHa3 + tmp_ac3;
			}
			else {
				UB3 = UB;
			}

			// nearest_segment = argmin( UB1, UB2, UB3 )
			if( UB1<=UB2 ) { if( UB1<=UB3 ) nearest_segment=1; else nearest_segment=3; }
			else { if( UB2<=UB3 ) nearest_segment=2; else nearest_segment=3; }
			
			switch( nearest_segment ) {
				case 1:
					aHa = tmp_aHa1;
					ac = tmp_ac1;
					for(i=0; i<dim; i++) {
						Ha[i] = Ha[i]*(1-gamma1) + gamma1*H(i,u);
						alpha[i] = alpha[i]*(1-gamma1);
					}
					alpha[u] += gamma1;
					break;

				case 2:
					aHa = tmp_aHa2;
					ac = tmp_ac2;
					tmp = alpha[v]*gamma2;
					for(i=0; i<dim; i++) 
						Ha[i] = Ha[i] + tmp*(H(i,u)-H(i,v));
					alpha[u] += tmp;
					alpha[v] -= tmp;
					break;

				case 3:
					aHa = tmp_aHa3;
					ac = tmp_ac3;
					tmp = alpha[v]*gamma3;
					for(i=0; i<dim; i++) {
						Ha[i] = gamma3*Ha[i] + H(i,u)*(1-gamma3+tmp) - tmp*H(i,v);
						alpha[i] = alpha[i]*gamma3;
					}
					alpha[u] += (1 - gamma3 + tmp);
					alpha[v] -= tmp;
					break;
			}
		}

		UB = 0.5*aHa + ac;
		min_beta = DBL_MAX;
		max_beta = -DBL_MAX;
		for(i=0; i<dim; i++) {
			beta = Ha[i] + c[i];
			if( alpha[i]!=0 && max_beta<beta ) {
				v = i;
				max_beta = beta;
			}
			if( beta<min_beta ) {
				u = i;
				min_beta = beta;
			}
		}

		LB = min_beta - 0.5*aHa;
		
		// Stopping conditions
		if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
		else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
		else if( LB > p_qp.threshLB )					exitflag = 3;
		else if( t >= p_qp.maxNbIter )					exitflag = 0;

		//printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);

		// Store selected values
//		if( t < History_size ) {
//			History[INDEX(0,t,2)] = LB;
//			History[INDEX(1,t,2)] = UB;
//		} 
//		else 
//			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n"); 

		t++;
	}

	// Print info about last iteration
	//printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB); 

	dual = UB;
	return exitflag;
}

int OptimizerNRBM::QP_kowalczyk(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp)
{
	int dim = H.getHeight();
	
	// Return values
	alpha.resize(1,dim);
	dual = 0;

	// Variables
	double LB,UB, aHa,ac, tmp,tmp1, beta,min_beta; 
	double x10,x11,x12,x20,x22, gamma,delta,tmp_UB,tmp_gamma,tmp_aHa,tmp_ac;
	int inx, i,t,exitflag; //,History_size;
	dVector Ha; //, History;

	// Initialization
	Ha.create(dim);
	//History_size = (p_qp.maxNbIter<HISTORY_BUF) ? p_qp.maxNbIter+1 : HISTORY_BUF;
	//History.create(History_size*2);

	tmp1 = DBL_MAX;
	for(i=0; i<dim; i++) {
		tmp = 0.5*diag_H[i] + c[i];
		if( tmp1 > tmp ) {
			tmp1 = tmp;
			inx = i;
		}
	}

	min_beta = DBL_MAX;
	for(i=0; i<dim; i++) {
		alpha[i] = 0;
		Ha[i] = H(i,inx);
		beta = Ha[i] + c[i];
		if( beta < min_beta ) {
			min_beta = beta; 
		}
	}
	
	alpha[inx] = 1;
	aHa = diag_H[inx];
	ac = c[inx];

	UB = 0.5*aHa + ac;
	LB = min_beta - 0.5*aHa;
	t = 0;
	//History[INDEX(0,0,2)] = LB;
	//History[INDEX(1,0,2)] = UB;

	//printf("QP_mdm(): init: UB=%f, LB=%f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB,LB,UB-LB,(UB-LB)/UB);

	// Stopping conditions
	if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
	else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
	else if( LB > p_qp.threshLB )					exitflag = 3;
	else											exitflag = -1;
	
	// Main QP Optimization Loop
	while( exitflag==-1 )
	{ 
		x11 = aHa;
		x10 = ac;
		
		// search for the rule that yields the biggest improvement
		for(i=0; i<dim; i++) {
			delta = Ha[i] + c[i] - aHa - ac;

			tmp_UB = DBL_MAX;
			if( delta<0 ) {
				// Kozinec rule
				x12 = Ha[i];
				x20 = c[i];
				x22 = diag_H[i];

				tmp_gamma = (x11-x12+x10-x20)/(x11-2*x12+x22);
				tmp_gamma = MIN(1,tmp_gamma);
				tmp_aHa = (1-tmp_gamma)*(1-tmp_gamma)*x11 + 2*(1-tmp_gamma)*tmp_gamma*x12 + tmp_gamma*tmp_gamma*x22;
				tmp_ac  = (1-tmp_gamma)*x10 + tmp_gamma*x20;
				tmp_UB  = 0.5*tmp_aHa + tmp_ac;
			}
			else if( delta>0 && alpha[i]<1 && alpha[i]>0 ) {
				x12 = (x11 - alpha[i]*Ha[i])/(1-alpha[i]);
				x22 = (x11 - 2*alpha[i]*Ha[i] + alpha[i]*alpha[i]*diag_H[i])/((1-alpha[i])*(1-alpha[i]));
				x20 = (x10 - alpha[i]*c[i])/(1-alpha[i]);

				tmp_gamma = (x11-x12+x10-x20)/(x11-2*x12+x22);
				tmp_gamma = MIN(1,tmp_gamma);
				tmp_aHa = (1-tmp_gamma)*(1-tmp_gamma)*x11 + 2*(1-tmp_gamma)*tmp_gamma*x12 + tmp_gamma*tmp_gamma*x22;
				tmp_ac  = (1-tmp_gamma)*x10 + tmp_gamma*x20;
				tmp_UB  = 0.5*tmp_aHa + tmp_ac;
			}

			if( tmp_UB<UB ) {
				UB = tmp_UB;
				gamma = tmp_gamma;
				aHa = tmp_aHa;
				ac = tmp_ac;
				inx = i;
			}
		}

		// Use the update with the biggest improvement
		delta = Ha[inx] + c[inx] - x11 - x10;
		if( delta<0 ) {
			// Kozinec rule
			for(i=0; i<dim; i++) {
				Ha[i] = Ha[i]*(1-gamma) + gamma*H(i,inx);
				alpha[i] = alpha[i]*(1-gamma);
			}
			alpha[inx] = alpha[inx] + gamma;
		}
		else {
			// Inverse Kozinec rule
			tmp = gamma*alpha[inx];
			tmp1 = 1-alpha[inx];
			for(i=0; i<dim; i++) {
				Ha[i] = (Ha[i]*(tmp+tmp1) - tmp*H(i,inx))/tmp1;
				alpha[i] = alpha[i]*(1-gamma) + gamma*alpha[i]/tmp1;
			}
			alpha[inx] = alpha[inx] - tmp/tmp1;
		}
		
		min_beta = DBL_MAX;
		for(i=0; i<dim; i++) {
			beta = Ha[i] + c[i];
			if( beta<min_beta ) 
				min_beta = beta;
		}

		LB = min_beta - 0.5*aHa;

		// Stopping conditions
		if( UB-LB <= p_qp.absTolerance )				exitflag = 1;
		else if( UB-LB <= fabs(UB)*p_qp.relTolerance )	exitflag = 2;
		else if( LB > p_qp.threshLB )					exitflag = 3;
		else if( t >= p_qp.maxNbIter )					exitflag = 0;

		//printf("QP_mdm():   %d: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", t, UB, LB, UB-LB, (UB-LB)/UB);

		// Store selected values
//		if( t < History_size ) {
//			History[INDEX(0,t,2)] = LB;
//			History[INDEX(1,t,2)] = UB;
//		} 
//		else 
//			printf("QP_mdn(): WARNING: History() is too small. Won't be logged\n"); 

		t++;
	}

	// Print info about last iteration
	//printf("QP_mdm():   exit: UB=%f, LB=f, UB-LB=%f, (UB-LB)/|UB|=%f\n", UB, LB, UB-LB, (UB-LB)/UB); 

	dual = UB;
	return exitflag;
}

int OptimizerNRBM::QP_kozinec(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp)
{
	return 0;
}