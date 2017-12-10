//-------------------------------------------------------------
// Hidden Conditional Random Field Library - Optimizer
// Component
//
//	January 30, 2006

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

//hCRF Library includes
#include "dataset.h"
#include "model.h"
#include "gradient.h"
#include "evaluator.h"
#include "matrix.h"

//#include "asa.h"
//#include "asa_usr_asa.h"
//#include "asa_usr.h"
// We derive from OWLQN, so we have to include the header 
#ifdef USEOWL
	#include "OWLQN.h"
#endif
//#ifdef USELBFGS
//	#include "lbfgs.h"
//#endif

class Optimizer {
public:
   Optimizer();
   virtual ~Optimizer();
//	char* getName()=0;
   virtual void optimize(Model* m, DataSet* X,
                         Evaluator* eval, Gradient* grad);
   virtual void optimize(Model* m, DataSet* X,
                         Evaluator* eval, GradientPerceptron* grad);
   virtual void setMaxNumIterations(int maxiter); 
   virtual int getMaxNumIterations();
   virtual int getLastNbIterations();
   virtual double getLastFunctionError();
   virtual double getLastNormGradient();

protected:
   int maxit; 
   void setConvergenceTolerance(double tolerance);
   int lastNbIterations;
   double lastFunctionError;
   double lastNormGradient;
};

class OptimizerCG: public Optimizer
{
public:
   OptimizerCG();
   ~OptimizerCG();
   virtual void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);
protected:
   static double callbackComputeError(double* weights);
   static void callbackComputeGradient(double* gradient, double* weights);
};



enum typeOptimizer
{
   optimBFGS = 0,
   optimDFP,
   optimFR,
   optimFRwithReset,
   optimPR,
   optimPRwithReset,
   optimPerceptronInitZero
};

class UnconstrainedOptimizer;


class OptimizerUncOptim: public Optimizer
{
public:
   ~OptimizerUncOptim();
   OptimizerUncOptim(typeOptimizer defaultOptimizer = optimBFGS);
   OptimizerUncOptim(const OptimizerUncOptim&);
   OptimizerUncOptim& operator=(const OptimizerUncOptim&){
        throw std::logic_error("Optimizer should not be copied");
    }
   void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);

private:
   UnconstrainedOptimizer* internalOptimizer;
   typeOptimizer optimizer;

};

struct USER_DEFINES;
typedef long int ALLOC_INT;

class OptimizerASA: public Optimizer
{
public:
    OptimizerASA();
    ~OptimizerASA();
    OptimizerASA(const OptimizerASA&);
    OptimizerASA& operator=(const OptimizerASA&){
        throw std::logic_error("Optimizer should not be copied");
    }
    virtual void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);
	
protected:
    // Protected, no need to have the definition here (we want to avoid including asas)
    static double callbackComputeError(double* weights,
                                       double *parameter_lower_bound,
                                       double *parameter_upper_bound,
                                       double *cost_tangents,
                                       double *cost_curvature,
                                       ALLOC_INT * parameter_dimension,
                                       int *parameter_int_real,
                                       int *cost_flag, int *exit_code,
                                       USER_DEFINES * USER_OPTIONS);
    static void callbackComputeGradient(double* gradient, double* weights);


private:
    double *parameter_lower_bound,  *parameter_upper_bound,  *cost_parameters;
    double *cost_tangents,  *cost_curvature;
    double cost_value;
    int *exit_code;
    USER_DEFINES *USER_OPTIONS;
    long int *rand_seed;
    int *parameter_int_real;
    long int *parameter_dimension;
    long int n_param;
    int initialize_parameters_value;
    int *cost_flag;
    static double (*rand_func_ptr)(long int *);
    static double randflt (long int * rand_seed);
    static double resettable_randflt (long int * rand_seed, int reset);
    static double myrand (long int * rand_seed);
    static int initialize_parameters(double *cost_parameters,
                                     double* parameter_lower_bound,
                                     double *parameter_upper_bound,
                                     long int *parameter_dimension,
                                     int *parameter_int_real);
    static Model* currentModel;
    static DataSet* currentDataset;
    static Evaluator* currentEvaluator;

};

#ifdef USEOWL
class OptimizerOWL: public Optimizer,DifferentiableFunction
{
public:
   ~OptimizerOWL();
   OptimizerOWL();
   OptimizerOWL(const OptimizerOWL&);
   OptimizerOWL& operator=(const OptimizerOWL&){
        throw std::logic_error("Optimizer should not be copied");
   };
   void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);   

protected:
   double Eval(const DblVec& input, DblVec& gradient);
private:
   Model* currentModel;
   DataSet* currentDataset;
   Evaluator* currentEvaluator;
   Gradient* currentGradient;
//   typeOptimizer optimizer;
   dVector vecGradient;
   OWLQN opt;
   DifferentiableFunction *obj;
};
#endif

#ifdef USELBFGS
typedef double lbfgsfloatval_t;
class OptimizerLBFGS: public Optimizer
{
  public:
    ~OptimizerLBFGS();
    OptimizerLBFGS();
    OptimizerLBFGS(const OptimizerLBFGS&);
    OptimizerLBFGS& operator=(const OptimizerLBFGS&){
        throw std::logic_error("Optimizer should not be copied");
    }
    void optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad);
    
  protected:
    static lbfgsfloatval_t _evaluate( void *instance, const lbfgsfloatval_t *x,
                                      lbfgsfloatval_t *g, const int n,
                                      const lbfgsfloatval_t)
    {
        return reinterpret_cast<OptimizerLBFGS*>(instance)->Eval(x, g, n);
    }

    static int _progress( void *instance, const lbfgsfloatval_t *x,
                          const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                          const lbfgsfloatval_t xnorm,
                          const lbfgsfloatval_t gnorm,
                          const lbfgsfloatval_t step, int n, int k, int ls )
    {
        return reinterpret_cast<OptimizerLBFGS*>(instance)->progress(x, g, fx,
                                                                     xnorm,
                                                                     gnorm,
                                                                     step, n
                                                                     , k, ls);
    }
    
    double Eval(const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n);
    int progress( const lbfgsfloatval_t *x, const lbfgsfloatval_t *g,
                  const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm,
                  const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
                  int n, int k, int ls);

  private:
    Model* currentModel;
    DataSet* currentDataset;
    Evaluator* currentEvaluator;
    Gradient* currentGradient;
    typeOptimizer optimizer;
    dVector vecGradient;
};
#endif

class OptimizerPerceptron: public Optimizer
{
public:
   OptimizerPerceptron(typeOptimizer defaultOptimizer = optimPerceptronInitZero);
   ~OptimizerPerceptron();
   virtual void optimize(Model* m, DataSet* X,Evaluator* eval, GradientPerceptron* grad);
private:
	Model* currentModel;
	DataSet* currentDataset;
	Evaluator* currentEvaluator;
	GradientPerceptron* currentGradient;  
    typeOptimizer optimizer;
	dVector vecGradient; // Is it useful?
};
 
class OptimizerNRBM: public Optimizer
{
	
public:
	OptimizerNRBM();
	~OptimizerNRBM();
	void optimize(Model* m, DataSet* X, Evaluator* eval, Gradient* grad);

private:
	struct nrbm_hyper_params {
		bool bRpositive;	// Tell if R(w) is always positive
		bool bRconvex;		// Tell if R(W) is convex
		bool bComputeGapQP;	// Verity the solution of approximation
		bool bLineSearch;	// Activate line search
		bool bCPMinApprox;	// Build cutting plane at minimier of approx. problem
		int maxNbIter;		// max number of iteration
		int maxNbCP;		// max number of cutting plane in approx. problem (NRBM working memory = maxNbCP * dim)
		double epsilon;		// relative tolerance (wrt. value of f)
	};

	struct wolfe_params {
		int maxNbIter;
		double a0;
		double a1;
		double c1;		// constant for sufficient reduction (Wolfe condition 1), 
		double c2;		// constant for curvature condition (Wolfe condition 2). 
		double amax;	// maximum step size allowed
	};

	struct reg_params {
		double lambda;	// regularization factor. should be m->regFactor(L1 or L2)
		dVector wreg;	// regularization point
		dVector reg;	// regularization weight
	};

	struct qp_params {
		const char* solver;	 // Solver to be used: {mdm|imdm|iimdm|kozinec|kowalczyk|keerthi}.
		int maxNbIter;		 // Maximal number of iterations (default inf).
		double absTolerance; // Absolute tolerance stopping condition (default 0.0).
		double relTolerance; // Relative tolerance stopping condition (default 1e-6).
		double threshLB;	 // Thereshold on the lower bound (default inf).
	};
 
	Model*		m_Model;
	DataSet*	m_Dataset;
	Evaluator*	m_Evaluator;
	Gradient*	m_Gradient;  

	void NRBM(
		dVector w0, 
		dVector& wbest, 
		const reg_params p_reg, 
		const nrbm_hyper_params p_nrbm);

	// Basic non-convex regularized bundle method for solving the unconstrained problem
	// min_w 0.5 lambda ||w||^2 + R(w)
	void NRBM_kernel(
		dVector w0, 
		dVector& wnbest, 
		const reg_params p_reg, 
		const nrbm_hyper_params p_nrbm); 

	void FGgrad_objective( 
		dVector w,  
		double& fval,
		dVector& grad,
		const reg_params p_reg);

	void fnew(
		dVector wnew, 
		double& Remp,
		dVector& gradw,
		const reg_params p_reg);

	void Fgrad(
		dVector w,  
		double& F,
		dVector& grad,
		const reg_params p_reg);

	void minimize_QP(
		double lambda,
		dMatrix Q,
		dVector B,
		bool Rpositive,
		double EPS,
		dVector& alpha,
		double& dual);

	
	int QP_kernel(dMatrix H, dVector c, dVector& alpha, double& dual, qp_params p_qp);
	int QP_mdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp);
	int QP_imdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp);
	int QP_iimdm(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp);
	int QP_kowalczyk(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp);
	int QP_keerthi(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp);
	int QP_kozinec(dMatrix H, dVector diag_H, dVector c, dVector& alpha, double& dual, qp_params p_qp);

	//
	// Predicate for sorting std::pair<int,int> wrt. the pair.second() in descending order
	static bool Desc(std::pair<int,int> a, std::pair<int,int> b) { return (a.second>b.second); };
	
	int myLineSearchWolfe( 
		dVector x0,		// current parameter
		double f0,		// function value at step size 0 
		dVector g0,		// gradient of f wrt its parameter at step size 0
		dVector s0,		// search direction at step size 0 
		double a1,		// initial step size to start 
		const reg_params p_reg,
		const wolfe_params p_wolfe,
		double& astar,	
		dVector& xstar, 
		double& fstar, 
		dVector& gstar,	
		dVector& x1, 
		double& f1, 
		dVector& g1);

	int myLineSearchZoom(
		double alo,
		double ahi,
		dVector x0,
		double f0,
		dVector g0,
		dVector s0,
		const reg_params p_reg,
		const wolfe_params p_wolfe,
		double linegrad0,
		double falo,
		double galo,
		double fhi,
		double ghi,
		double& astar, 
		dVector& xstar,
		double& fstar,
		dVector& gstar);



	// INPUT
    //		H	[M x N] matrix
    //		a	[D x 1] vector
    //		idx	[D x 1] vector
    // OUTPUT
    //		w	[M x 1] vector
	void w_sum_row(dMatrix H, dVector a, std::list<int> idx, dVector& w);

	// INPUT
    //		H	[M x N] matrix
    //		a	[D x 1] vector
    //		idx [D x 1] vector
    // OUTPUT
    //		w = sum( H .* repmat(a,1,N));
	void w_sum_col(dMatrix H, dVector a, std::list<int> idx, dVector& w);


	// SETDIFF(A,B) when A and B are vectors returns the values in A that are not in B.  
	// The result will be sorted.
	std::list<int> matlab_setdiff(std::list<int> list_a, std::list<int> list_b); 
};
 


#endif



