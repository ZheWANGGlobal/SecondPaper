#ifndef _ASA_USER_H_
#define _ASA_USER_H_
#ifdef __cplusplus
extern "C" {
#endif

/***********************************************************************
* Adaptive Simulated Annealing (ASA)
* Lester Ingber <ingber@ingber.com>
* Copyright (c) 1993-2007 Lester Ingber.  All Rights Reserved.
* The ASA-LICENSE file must be included with ASA code.
***********************************************************************/

  /* $Id: asa_usr.h,v 26.23 2007/01/31 20:13:30 ingber Exp ingber $ */

  /* asa_usr.h for Adaptive Simulated Annealing */

#include "asa_usr_asa.h"

#define SHUFFLE 256             /* size of random array */

#if ASA_TEMPLATE_ASA_OUT_PID
#include <sys/types.h>
#endif

#if TIME_CALC
  /* print the time every PRINT_FREQUENCY function evaluations
     Define PRINT_FREQUENCY to 0 to not print out the time. */
#define PRINT_FREQUENCY ((LONG_INT) 1000)
#endif

#if USER_ACCEPTANCE_TEST
#define MIN(x,y)	((x) < (y) ? (x) : (y))
#endif

  /* system function prototypes */

#if ASA_TEMPLATE_ASA_OUT_PID
  int getpid ();
#endif

#if HAVE_ANSI

/* This block gives trouble under some Ultrix */
#if FALSE
#if OPTIONS_FILE
  int fscanf (FILE * fp, char *string, ...);
#endif
#endif

#if IO_PROTOTYPES
#if OPTIONS_FILE
  int fscanf ();
#endif
#endif

  /* user-defined */
  double USER_COST_FUNCTION (double *cost_parameters,
                             double *parameter_lower_bound,
                             double *parameter_upper_bound,
                             double *cost_tangents,
                             double *cost_curvature,
                             ALLOC_INT * parameter_dimension,
                             int *parameter_int_real,
                             int *cost_flag,
                             int *exit_code, USER_DEFINES * USER_OPTIONS);
#if ASA_LIB
  int asa_main (
#if ASA_TEMPLATE_LIB
                 double *main_cost_value,
                 double *main_cost_parameters, int *main_exit_code
#endif
    );
#else
  int main (int argc, char **argv);
#endif

#if ASA_TEMPLATE_LIB
  int main ();
#endif

  /* possibly with accompanying data file */
  int initialize_parameters (double *cost_parameters,
                             double *parameter_lower_bound,
                             double *parameter_upper_bound,
                             double *cost_tangents,
                             double *cost_curvature,
                             ALLOC_INT * parameter_dimension,
                             int *parameter_int_real,
#if OPTIONS_FILE_DATA
                             FILE * ptr_options,
#endif
                             USER_DEFINES * USER_OPTIONS);

#if ASA_LIB
  LONG_INT asa_seed (LONG_INT seed);
#endif
  double myrand (LONG_INT * rand_seed);
  double randflt (LONG_INT * rand_seed);
  double resettable_randflt (LONG_INT * rand_seed, int reset);

#if USER_COST_SCHEDULE
  double user_cost_schedule (double test_temperature,
                             USER_DEFINES * USER_OPTIONS);
#endif

#if USER_ACCEPTANCE_TEST
  void user_acceptance_test (double current_cost,
                             double *parameter_lower_bound,
                             double *parameter_upper_bound,
                             ALLOC_INT * parameter_dimension,
                             USER_DEFINES * USER_OPTIONS);
#endif

#if USER_GENERATING_FUNCTION
  double user_generating_distrib (LONG_INT * seed,
                                  ALLOC_INT * parameter_dimension,
                                  ALLOC_INT index_v,
                                  double temperature_v,
                                  double init_param_temp_v,
                                  double temp_scale_params_v,
                                  double parameter_v,
                                  double parameter_range_v,
                                  double *last_saved_parameter,
                                  USER_DEFINES * USER_OPTIONS);
#endif

#if USER_REANNEAL_COST
  int user_reanneal_cost (double *cost_best,
                          double *cost_last,
                          double *initial_cost_temperature,
                          double *current_cost_temperature,
                          USER_DEFINES * USER_OPTIONS);
#endif

#if USER_REANNEAL_PARAMETERS
  double user_reanneal_params (double current_temp,
                               double tangent,
                               double max_tangent,
                               USER_DEFINES * USER_OPTIONS);
#endif

#if ASA_TEMPLATE_SAMPLE
  void sample (FILE * ptr_out, FILE * ptr_asa);
#endif

  void Exit_USER (char *statement);

#else                           /* HAVE_ANSI */

#if IO_PROTOTYPES
#if OPTIONS_FILE
  int fscanf ();
#endif
#endif

  /* user-defined */
  double USER_COST_FUNCTION ();
#if ASA_LIB
  int asa_main ();
#else
  int main ();
#endif

#if ASA_TEMPLATE_LIB
  int main ();
#endif

  int initialize_parameters (); /* possibly with accompanying
                                   data file */
#if ASA_LIB
  LONG_INT asa_seed ();
#endif
  double myrand ();
  double randflt ();
  double resettable_randflt ();

#if USER_COST_SCHEDULE
  double user_cost_schedule ();
#endif

#if USER_ACCEPTANCE_TEST
  void user_acceptance_test ();
#endif

#if USER_GENERATING_FUNCTION
  double user_generating_distrib ();
#endif

#if USER_REANNEAL_COST
  int user_reanneal_cost ();
#endif

#if USER_REANNEAL_PARAMETERS
  double user_reanneal_params ();
#endif

#if ASA_TEMPLATE_SAMPLE
  void sample ();
#endif

  void Exit_USER ();

#endif                          /* HAVE_ANSI */

#if SELF_OPTIMIZE
#if TIME_CALC
#define RECUR_PRINT_FREQUENCY ((LONG_INT) 1)
#endif

#if HAVE_ANSI                   /* HAVE_ANSI SELF_OPTIMIZE */
  double RECUR_USER_COST_FUNCTION (double *recur_cost_parameters,
                                   double *recur_parameter_lower_bound,
                                   double *recur_parameter_upper_bound,
                                   double *recur_cost_tangents,
                                   double *recur_cost_curvature,
                                   ALLOC_INT * recur_parameter_dimension,
                                   int *recur_parameter_int_real,
                                   int *recur_cost_flag,
                                   int *recur_exit_code,
                                   USER_DEFINES * RECUR_USER_OPTIONS);

  int recur_initialize_parameters (double *recur_cost_parameters,
                                   double *recur_parameter_lower_bound,
                                   double *recur_parameter_upper_bound,
                                   double *recur_cost_tangents,
                                   double *recur_cost_curvature,
                                   ALLOC_INT * recur_parameter_dimension,
                                   int *recur_parameter_int_real,
#if RECUR_OPTIONS_FILE_DATA
                                   FILE * recur_ptr_options,
#endif
                                   USER_DEFINES * RECUR_USER_OPTIONS);

#if USER_COST_SCHEDULE
  double recur_user_cost_schedule (double test_temperature,
                                   USER_DEFINES * RECUR_USER_OPTIONS);
#endif

#if USER_ACCEPTANCE_TEST
  void recur_user_acceptance_test (double current_cost,
                                   double *recur_parameter_lower_bound,
                                   double *recur_parameter_upper_bound,
                                   ALLOC_INT * recur_parameter_dimension,
                                   USER_DEFINES * RECUR_USER_OPTIONS);
#endif

#if USER_GENERATING_FUNCTION
  double recur_user_generating_distrib (LONG_INT * seed,
                                        ALLOC_INT * recur_parameter_dimension,
                                        ALLOC_INT index_v,
                                        double temperature_v,
                                        double init_param_temp_v,
                                        double temp_scale_params_v,
                                        double parameter_v,
                                        double parameter_range_v,
                                        double *last_saved_parameter,
                                        USER_DEFINES * RECUR_USER_OPTIONS);
#endif

#if USER_REANNEAL_COST
  int recur_user_reanneal_cost (double *cost_best,
                                double *cost_last,
                                double *initial_cost_temperature,
                                double *current_cost_temperature,
                                USER_DEFINES * RECUR_USER_OPTIONS);
#endif

#if USER_REANNEAL_PARAMETERS
  double recur_user_reanneal_params (double current_temp,
                                     double tangent,
                                     double max_tangent,
                                     USER_DEFINES * RECUR_USER_OPTIONS);
#endif

#else                           /* HAVE_ANSI SELF_OPTIMIZE */

  double RECUR_USER_COST_FUNCTION ();
  int recur_initialize_parameters ();

#if USER_COST_SCHEDULE
  double recur_user_cost_schedule ();
#endif

#if USER_ACCEPTANCE_TEST
  void recur_user_acceptance_test ();
#endif

#if USER_GENERATING_FUNCTION
  double recur_user_generating_distrib ();
#endif

#if USER_REANNEAL_COST
  int recur_user_reanneal_cost ();
#endif

#if USER_REANNEAL_PARAMETERS
  double recur_user_reanneal_params ();
#endif

#endif                          /* HAVE_ANSI */
#endif                          /* SELF_OPTIMIZE */

#if FITLOC
#if HAVE_ANSI
  double
    calcf (double (*user_cost_function)

            
           (double *, double *, double *, double *, double *, ALLOC_INT *,
            int *, int *, int *, USER_DEFINES *), double *cost_parameters,
           double *parameter_lower_bound, double *parameter_upper_bound,
           double *cost_tangents, double *cost_curvature,
           ALLOC_INT * parameter_dimension, int *parameter_int_real,
           int *cost_flag, int *exit_code, USER_DEFINES * USER_OPTIONS,
           FILE * ptr_out);

  double
    fitloc (double (*user_cost_function)

             
            (double *, double *, double *, double *, double *, ALLOC_INT *,
             int *, int *, int *, USER_DEFINES *), double *cost_parameters,
            double *parameter_lower_bound, double *parameter_upper_bound,
            double *cost_tangents, double *cost_curvature,
            ALLOC_INT * parameter_dimension, int *parameter_int_real,
            int *cost_flag, int *exit_code, USER_DEFINES * USER_OPTIONS,
            FILE * ptr_out);

  int
    simplex (double (*user_cost_function)

              
             (double *, double *, double *, double *, double *, ALLOC_INT *,
              int *, int *, int *, USER_DEFINES *), double *cost_parameters,
             double *parameter_lower_bound, double *parameter_upper_bound,
             double *cost_tangents, double *cost_curvature,
             ALLOC_INT * parameter_dimension, int *parameter_int_real,
             int *cost_flag, int *exit_code, USER_DEFINES * USER_OPTIONS,
             FILE * ptr_out, double tol1, double tol2, int no_progress,
             double alpha, double beta1, double beta2, double gamma,
             double delta);
#else                           /* HAVE_ANSI */

  double calcf ();
  double fitloc ();
  int simplex ();

#endif                          /* HAVE_ANSI */
#endif                          /* FITLOC */

#ifdef __cplusplus
}
#endif
#endif                          /* _ASA_USER_H_ */
