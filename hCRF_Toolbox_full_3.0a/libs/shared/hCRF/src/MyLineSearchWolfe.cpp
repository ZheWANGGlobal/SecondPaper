//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Implementation of
// Line search method satisfying strong Wolfe condition.
//  
// Determines a line search step size that satisfies the strong 
// Wolfe condition. The algorithm implements what was described 
// in Numerial Optimization (Nocedal and Wright, Springer 1999), 
// Algol 3.5 (Line Search Algorithm) and Algol. 3.6 (Zoom).
//
// Inputs:
//	lambda: regularization factor for FGgrad_objective(...)
//      x0: current parameter
//      f0: function value at step size 0 
//      g0: gradient of f with respect to its parameter at step size 0
//      s0: search direction at step size 0 
//      a1: initial step size to start
//       p: wolfe constants
// 	  amax:  maximum step size allowed
// 	    c1:  the constant for sufficient reduction (Wolfe condition 1), 
//      c2:  the constant for curvature condition (Wolfe condition 2).
// maxiter: maxium iteration to search for ste size
//
// Outputs
//   x1,f1,g1 are the solution, fval and grad correspondant to stepsize a1
//
// Yale Song (yalesong@csail.mit.edu)
// October, 2011

#include "optimizer.h"  
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define isnan(x) ((x) != (x))
 
int OptimizerNRBM::myLineSearchWolfe(
	dVector x0, double f0, dVector g0, dVector s0, double a1, 
	const reg_params p_reg, const wolfe_params p_wolfe,
	double &astar, dVector &xstar, double &fstar, dVector &gstar, 
	dVector &x1, double &f1, dVector &g1)
{	
	int numeval = 0; // FGgrad_objective() call count. Value to be returned
	
	dVector Wtmp; 

	double ai,ai_1,fi,fi_1;
	ai_1 = 0;
	ai = a1;		// initial step size to start
	fi_1 = f0;		// function value at step size 0

	double linegrad0, linegradi, linegradi_1;
	Wtmp.set(g0); Wtmp.eltMpy(s0); linegrad0 = Wtmp.sum();
	linegradi_1 = linegrad0;

	x1.set(s0); x1.multiply(ai); x1.add(x0); // x1 = x0 + ai*s0; 
	FGgrad_objective(x1,f1,g1,p_reg); numeval++; // save initial objective values

	for(int i=0;; i++) {
		xstar.set(s0); xstar.multiply(ai); xstar.add(x0); // xstar = x0 + ai*s0; 
		fi=0; dVector gi(x0.getLength()); 
		FGgrad_objective(xstar,fi,gi,p_reg); numeval++;
		
		Wtmp.set(gi); Wtmp.eltMpy(s0); 
		linegradi = Wtmp.sum();		
		if( fi>=fi_1 || fi>(f0+p_wolfe.c1*ai*linegrad0) ) {
			numeval += myLineSearchZoom(
						ai_1,ai,x0,f0,g0,s0,p_reg,p_wolfe,
						linegrad0,fi_1,linegradi_1,fi,linegradi,
						astar,xstar,fstar,gstar);
			return numeval;
		}
		
		if( fabs(linegradi)<=-p_wolfe.c2*linegrad0 ) {
			astar=ai; fstar=fi; gstar.set(gi);
			return numeval;
		}

		if( linegradi>=0 ) {
			numeval += myLineSearchZoom(
						ai,ai_1,x0,f0,g0,s0,p_reg,p_wolfe,
						linegrad0,fi,linegradi,fi_1,linegradi_1,
						astar,xstar,fstar,gstar);
			return numeval;
		}
		i++;
		if( fabs(ai-p_wolfe.amax)<=0.01*p_wolfe.amax || i>=p_wolfe.maxNbIter ) {
			astar=ai; fstar=fi; gstar.set(gi);
			return numeval;
		}
		ai_1 = ai;
		fi_1 = fi;
		linegradi_1 = linegradi;
		ai = (ai+p_wolfe.amax)/2;
	}
}

int OptimizerNRBM::myLineSearchZoom(
	double alo, double ahi, dVector x0, double f0, dVector g0, dVector s0, 
	const reg_params p_reg, const wolfe_params p_wolfe,
	double linegrad0, double falo, double galo, double fhi, double ghi,
	double &astar, dVector &xstar, double &fstar, dVector &gstar)
{
	int numeval = 0; // FGgrad_objective() call count. Value to be returned
	int i = 0;		 // if i>p_wolfe.maxNbIter halt

	dVector Wtmp; 

	double aj,fj;
	double d1,d2;
	double linegradj;

	for(int i=0;;i++) {
		d1 = galo+ghi - 3*(falo-fhi)/(alo-ahi);
		d2 = sqrt(MAX(0,d1*d1 - galo*ghi));
		aj = ahi - (ahi-alo)*(ghi+d2-d1)/(ghi-galo+2*d2);
		if( isnan(aj) ) aj=-DBL_MAX; // to avoid numerical error
		if( alo<ahi ) { 
			if( aj<alo || aj>ahi ) 
				aj = (alo+ahi)/2; 
		}
		else { 
			if( aj>alo || aj<ahi ) 
				aj = (alo+ahi)/2; 
		}
		xstar.set(s0); xstar.multiply(aj); xstar.add(x0); // xstar = x0 + aj*s0;
		fj=0; dVector gj(x0.getLength()); 
		FGgrad_objective(xstar,fj,gj,p_reg); numeval++;

		if( fj>falo || fj>(f0+p_wolfe.c1*aj*linegrad0) ) {
			ahi=aj; fhi=fj; 
			Wtmp.set(gj); Wtmp.eltMpy(s0); ghi=Wtmp.sum();
		}
		else {
			Wtmp.set(gj); Wtmp.eltMpy(s0); linegradj=Wtmp.sum();
			if( fabs(linegradj)<=-p_wolfe.c2*linegrad0 ) {
				astar=aj; fstar=fj; gstar.set(gj); 
				return numeval;
			}
			if( linegradj*(ahi-alo)>=0 ) {
				ahi=alo; fhi=falo; ghi=galo;
			}
			alo=aj; falo=fj; galo=linegradj;
		}
		
		if( fabs(alo-ahi)<=0.01*alo || i>=p_wolfe.maxNbIter ) {
			astar=aj; fstar=fj; gstar.set(gj);
			return numeval;
		}
	}
}
 