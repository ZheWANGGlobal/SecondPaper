#define USER_ID "/* $Id: asa_usr.c,v 26.23 2007/01/31 20:13:28 ingber Exp ingber $ */"

#define MULT ((LONG_INT) 25173)
#define MOD ((LONG_INT) 65536)
#define INCR ((LONG_INT) 13849)
#define FMOD ((double) 65536.0)
#define _CRT_SECURE_NO_WARNINGS


#include "optimizer.h"
#include "asa.h"
#include "asa_usr_asa.h"
#include "asa_usr.h"
#include <iostream>
#include <fstream>

//Initialize the static member of the class
Model* OptimizerASA::currentModel = NULL;
DataSet* OptimizerASA::currentDataset = NULL;
Evaluator* OptimizerASA::currentEvaluator = NULL;

// This function computes the actual cost function (Conditional Negative Log Maximum Likelihood)
double OptimizerASA::callbackComputeError(double* weights,
										  double *parameter_lower_bound,
										  double *parameter_upper_bound,
										  double *cost_tangents,
										  double *cost_curvature,
										  ALLOC_INT * parameter_dimension,
										  int *parameter_int_real,
										  int *cost_flag, int *exit_code, USER_DEFINES * USER_OPTIONS)
{
	// We set the weights in our model according to the wishes of OptimizerASA
	dVector vecWeights(currentModel->getWeights()->getLength());
	memcpy(vecWeights.get(),weights,currentModel->getWeights()->getLength()*sizeof(double));
	currentModel->setWeights(vecWeights);
	if(currentModel->getDebugLevel() >= 1){
		std::cout << "Compute error... "  << std::endl;
	}
	double errorVal = currentEvaluator->computeError(currentDataset, currentModel);
	if(currentModel->getDebugLevel() >= 1){
		std::cout << "erroval = "<<errorVal<<std::endl;
		if(currentModel->getDebugLevel() >= 3){
			std::cout<<"x: "<<vecWeights;
		}
	}
	return errorVal;	
}



//Empty, left for compliance
void OptimizerASA::callbackComputeGradient(double* gradient, double* weights)
{
}

#define _CRT_SECURE_NO_WARNINGS

//Constructor only receives the pointer to the cost function and the size of the
// parameter vector
OptimizerASA::OptimizerASA(): Optimizer()
{
}

OptimizerASA::~OptimizerASA()
{
  /*fclose(ptr_out);

  if (USER_OPTIONS->Curvature_0 == FALSE || USER_OPTIONS->Curvature_0 ==  - 1)
    free(cost_curvature);

  free(USER_OPTIONS);
  free(parameter_dimension);
  free(exit_code);
  free(cost_flag);
  free(parameter_lower_bound);
  free(parameter_upper_bound);
  free(cost_parameters);
  free(parameter_int_real);
  free(cost_tangents);
  free(rand_seed);*/

}


void OptimizerASA::optimize(Model* m, DataSet* X,Evaluator* eval, Gradient* grad)
{
	
	///////////////////////////////////////////
	//Initialize ASA Paraphernalia
	//////////////////////////////////////////
	if ((USER_OPTIONS = (USER_DEFINES*)calloc(1, sizeof(USER_DEFINES))) == NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): USER_DEFINES\n\n");
		return;
	}

	if ((rand_seed = (ALLOC_INT*)calloc(1, sizeof(ALLOC_INT))) == NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): rand_seed\n\n");
		return;
	}
	if ((parameter_dimension = (ALLOC_INT*)calloc(1, sizeof(ALLOC_INT))) == NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): parameter_dimension\n\n");
		return;
	}
	if ((exit_code = (int*)calloc(1, sizeof(int))) == NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): exit_code\n\n");
		return;
	}
	if ((cost_flag = (int*)calloc(1, sizeof(int))) == NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): cost_flag\n\n");
		return;
	}


	/* initialize random number generator with first call */
	*rand_seed = 696969;
	resettable_randflt(rand_seed, 1);

	/* Initialize the users parameters, allocating space, etc.
	Note that the default is to have asa generate the initial
	cost_parameters that satisfy the user's constraints. */


	USER_OPTIONS->Limit_Acceptances = 1000;
	USER_OPTIONS->Limit_Generated = 99999;
	USER_OPTIONS->Limit_Invalid_Generated_States = 1000;
	USER_OPTIONS->Accepted_To_Generated_Ratio = 1.0e-4;
	USER_OPTIONS->Cost_Precision = 1.0e-18;
	USER_OPTIONS->Maximum_Cost_Repeat = 5;
	USER_OPTIONS->Number_Cost_Samples = 5;
	USER_OPTIONS->Temperature_Ratio_Scale = 1.0e-5;
	USER_OPTIONS->Cost_Parameter_Scale_Ratio = 1.0;
	USER_OPTIONS->Temperature_Anneal_Scale = 100.0;
	USER_OPTIONS->Include_Integer_Parameters = 0;
	USER_OPTIONS->User_Initial_Parameters = 0;
	USER_OPTIONS->Sequential_Parameters = -1;
	USER_OPTIONS->Initial_Parameter_Temperature = 1.0;
	USER_OPTIONS->Acceptance_Frequency_Modulus = 100;
	USER_OPTIONS->Generated_Frequency_Modulus = 10000;
	USER_OPTIONS->Reanneal_Cost = 1;
	USER_OPTIONS->Reanneal_Parameters = 1;
	USER_OPTIONS->Delta_X = 0.001;
	USER_OPTIONS->User_Tangents = 0;
	USER_OPTIONS->Curvature_0 = 0;

	*parameter_dimension=m->getWeights()->getLength();


	/* allocate parameter minimum space */
	if ((parameter_lower_bound = (double*)calloc(*parameter_dimension, sizeof(double))) == NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): parameter_lower_bound\n\n");
		return;
	}
	/* allocate parameter maximum space */
	if ((parameter_upper_bound = (double*)calloc(*parameter_dimension, sizeof(double))) == NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): parameter_upper_bound\n\n");
		return;
	}
	/* allocate parameter initial values; the parameter final values
	will be stored here later */
	if ((cost_parameters = (double*)calloc(*parameter_dimension, sizeof(double)))== NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): cost_parameters\n\n");
		return;
	}
	/* allocate the parameter types, real or integer */	
	if ((parameter_int_real = (int*)calloc(*parameter_dimension, sizeof(int))) ==NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): parameter_int_real\n\n");
		return;
	}
	/* allocate space for parameter cost_tangents -
	used for reannealing */
	if ((cost_tangents = (double*)calloc(*parameter_dimension, sizeof(double)))== NULL)
	{
		printf("\n\n*** EXIT calloc failed *** main()/asa_main(): cost_tangents\n\n");
		return;
	}

	if (USER_OPTIONS->Curvature_0 == FALSE || USER_OPTIONS->Curvature_0 ==  - 1)
	{
		/* allocate space for parameter cost_curvatures/covariance */
		if ((cost_curvature = (double*)calloc((*parameter_dimension)*(*parameter_dimension), sizeof(double))) == NULL)
		{
			printf("\n\n*** EXIT calloc failed *** main()/asa_main(): cost_curvature\n\n");
			return;
		}
	}
	else
	{
		cost_curvature = (double*)NULL;
	}


  	#if QUENCH_PARAMETERS
		if ((USER_OPTIONS->User_Quench_Param_Scale = (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) 
		{
			printf("\n\n*** EXIT calloc failed *** initialize_parameters(): USER_OPTIONS->User_Quench_Cost_Scale\n\n");
			return;
		}
		//Initialize QUENCH parameters
		for (int index = 0; index <  *parameter_dimension; ++index)
			USER_OPTIONS->User_Quench_Cost_Scale[index]=1.0;
	#endif

   #if QUENCH_COST
		if ((USER_OPTIONS->User_Quench_Cost_Scale = (double *) calloc (*parameter_dimension, sizeof (double))) == NULL) 
		{
			printf("\n\n*** EXIT calloc failed *** initialize_parameters(): USER_OPTIONS->User_Quench_Param_Scale\n\n");
			return;
		}
		//Initialize QUENCH parameters
		for (int index = 0; index <  *parameter_dimension; ++index)
			USER_OPTIONS->User_Quench_Cost_Scale[index]=1.0;
	#endif


	initialize_parameters_value = initialize_parameters(cost_parameters,parameter_lower_bound, parameter_upper_bound, parameter_dimension, parameter_int_real);

	if (initialize_parameters_value ==  - 2)
		return;
		


	///////////////////////////////////////////
	//End initialization ASA Paraphernalia
	//////////////////////////////////////////

	currentModel = m;
	currentDataset = X;
	currentEvaluator = eval;
	double* weights;
	int status = -1;	

	weights = (double *) malloc (currentModel->getWeights()->getLength()*sizeof(double)) ;
	memcpy(weights,currentModel->getWeights()->get(),currentModel->getWeights()->getLength()*sizeof(double));

	// Call
	status = (int)asa(&callbackComputeError,&randflt, rand_seed, weights,
						parameter_lower_bound, parameter_upper_bound, cost_tangents, cost_curvature,
						parameter_dimension, parameter_int_real, cost_flag, exit_code, USER_OPTIONS);


	dVector vecGradient(currentModel->getWeights()->getLength());
	memcpy(vecGradient.get(),weights,currentModel->getWeights()->getLength()*sizeof(double));
	currentModel->setWeights(vecGradient);


	if (*exit_code ==  - 1)
	{
		printf("\n\n*** error in calloc in ASA ***\n\n");
		return;
	}

	if(currentModel->getDebugLevel() >= 1)
	{
		printf("exit code = %d\n",  *exit_code);
		printf("final cost value = %12.7g\n", cost_value);
		printf("parameter\tvalue\n");
	}
	if(currentModel->getDebugLevel() >= 3)
	{
		for (n_param = 0; n_param <  *parameter_dimension; ++n_param)
		{
			printf("%ld\t\t%12.7g\n", n_param, cost_parameters[n_param]);
		}
	}
}



int OptimizerASA::initialize_parameters(double *cost_parameters, double
  *parameter_lower_bound, double *parameter_upper_bound, ALLOC_INT *parameter_dimension, 
  int *parameter_int_real)

{
 //	Real initialization
  for (int index = 0; index <  *parameter_dimension; ++index)
  {
	// Upper and lower initial values
	parameter_lower_bound[index]=-1;
	parameter_upper_bound[index]=1;
	//Initial value
	cost_parameters[index]=0.5;
	// Initialize as "real" numbers
	parameter_int_real[index]=-1;

  }

  return (0);
}



double OptimizerASA::randflt (LONG_INT * rand_seed)
{
  return (resettable_randflt (rand_seed, 0));
}

double OptimizerASA::resettable_randflt (LONG_INT * rand_seed, int reset)
{
  double rranf;
  unsigned kranf;
  int n;
  static int initial_flag = 0;
  LONG_INT initial_seed;
  static double random_array[SHUFFLE];  /* random variables */
  
  if (*rand_seed < 0)
    *rand_seed = -*rand_seed;

  if ((initial_flag == 0) || reset) 
  {
    initial_seed = *rand_seed;

  	for (n = 0; n < SHUFFLE; ++n)
  	  random_array[n] = myrand (&initial_seed);
  	
  	initial_flag = 1;
  
  	for (n = 0; n < 1000; ++n)  /* warm up random generator */
  	  rranf = randflt (&initial_seed);

    rranf = randflt (rand_seed);

    return (rranf);
  }

  kranf = (unsigned) (myrand (rand_seed) * SHUFFLE) % SHUFFLE;
  rranf = *(random_array + kranf);
  *(random_array + kranf) = myrand (rand_seed);

  return (rranf);
  
}

/***********************************************************************
* double myrand - returns random number between 0 and 1
*	This routine returns the random number generator between 0 and 1
***********************************************************************/
double OptimizerASA::myrand (LONG_INT * rand_seed)
{
  *rand_seed = (LONG_INT) ((MULT * (*rand_seed) + INCR) % MOD);
  return ((double) (*rand_seed) / FMOD);
}


