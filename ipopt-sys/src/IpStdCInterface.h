/*************************************************************************
   Copyright (C) 2004, 2010 International Business Machines and others.
   All Rights Reserved.
   This code is published under the Eclipse Public License.
 
   $Id: IpStdCInterface.h 2082 2012-02-16 03:00:34Z andreasw $
 
   Authors:  Carl Laird, Andreas Waechter     IBM    2004-09-02
   Edited by: Egor Larionov (2018)
 *************************************************************************/

#ifndef __IPSTDCINTERFACE_H__
#define __IPSTDCINTERFACE_H__

#ifndef IPOPT_EXPORT
#ifdef _MSC_VER
#ifdef IPOPT_DLL
#define IPOPT_EXPORT(type) __declspec(dllexport) type __cdecl
#else
#define IPOPT_EXPORT(type) type __cdecl
#endif
#else 
#define IPOPT_EXPORT(type) type
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /** Type for all number.  We need to make sure that this is
      identical with what is defined in Common/IpTypes.hpp */
  typedef double Number;

  /** Type for all incides.  We need to make sure that this is
      identical with what is defined in Common/IpTypes.hpp */
  typedef int Index;

  /** Type for all integers.  We need to make sure that this is
      identical with what is defined in Common/IpTypes.hpp */
  typedef int Int;

  /* This includes the SolverReturn enum type */
#include "IpReturnCodes.h"

  /** Structure collecting all information about the problem
   *  definition and solve statistics etc.  This is defined in the
   *  source file. */
  struct IpoptProblemInfo;

  /** Pointer to a Ipopt Problem. */
  typedef struct IpoptProblemInfo* IpoptProblem;

  /** define a boolean type for C */
  typedef int Bool;
#ifndef TRUE
# define TRUE (1)
#endif
#ifndef FALSE
# define FALSE (0)
#endif

  /** A pointer for anything that is to be passed between the called
   *  and individual callback function */
  typedef void * UserDataPtr;

  /** Type defining the callback function for setting scaling
   *  parameters. This method is called if nlp_scaling_method is set
   *  to user-scaling. This function is optional. */
  typedef Bool (*ScalingParams_CB)(Number* obj_scaling,
                                   Bool* use_x_scaling, Index n,
                                   Number* x_scaling,
                                   Bool* use_g_scaling, Index m,
                                   Number* g_scaling,
                                   UserDataPtr user_data);

  /** Type defining the callback function for setting sizes for arrays
   *  that will store variables, constraint values and derivatives. */
  typedef Bool (*Sizes_CB)(Index *n, Index *m,
                           Index *nnz_jac_g, Index *nnz_h_lag,
                           UserDataPtr user_data);

  /** Type defining the callback function for initializing variables
   *  and multipliers. */
  typedef Bool (*Init_CB)(Index n, Bool init_x, Number* x, 
                          Bool init_z, Number* z_L, Number* z_U, 
                          Index m, Bool init_lambda, Number* lambda, 
                          UserDataPtr user_data);

  /** Type defining the callback function for specifying variable and
   *  constraint lower and upper bounds. */
  typedef Bool (*Bounds_CB)(Index n, Number* x_l, Number* x_u,
                            Index m, Number* g_l, Number* g_u,
                            UserDataPtr user_data);

  /** Type defining the callback function for evaluating the value of
   *  the objective function.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_F_CB)(Index n, const Number* x, Bool new_x,
                            Number* obj_value, UserDataPtr user_data);

  /** Type defining the callback function for evaluating the gradient of
   *  the objective function.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_Grad_F_CB)(Index n, const Number* x, Bool new_x,
                                 Number* grad_f, UserDataPtr user_data);

  /** Type defining the callback function for evaluating the value of
   *  the constraint functions.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_G_CB)(Index n, const Number* x, Bool new_x,
                            Index m, Number* g, UserDataPtr user_data);

  /** Type defining the callback function for evaluating the Jacobian of
   *  the constrant functions.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_Jac_G_CB)(Index n, const Number *x, Bool new_x,
                                Index m, Index nele_jac,
                                Index *iRow, Index *jCol, Number *values,
                                UserDataPtr user_data);

  /** Type defining the callback function for evaluating the Hessian of
   *  the Lagrangian function.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_H_CB)(Index n, const Number *x, Bool new_x, Number obj_factor,
                            Index m, const Number *lambda, Bool new_lambda,
                            Index nele_hess, Index *iRow, Index *jCol,
                            Number *values, UserDataPtr user_data);

  /** Type defining the callback function for giving intermediate
   *  execution control to the user.  If set, it is called once per
   *  iteration, providing the user with some information on the state
   *  of the optimization.  This can be used to print some
   *  user-defined output.  It also gives the user a way to terminate
   *  the optimization prematurely.  If this method returns false,
   *  Ipopt will terminate the optimization. */
  typedef Bool (*Intermediate_CB)(Index alg_mod, /* 0 is regular, 1 is resto */
				  Index iter_count, Number obj_value,
				  Number inf_pr, Number inf_du,
				  Number mu, Number d_norm,
				  Number regularization_size,
				  Number alpha_du, Number alpha_pr,
				  Index ls_trials, UserDataPtr user_data);

  enum CreateProblemStatus {
      Success,
      MissingSizes,
      MissingInitialGuess,
      MissingBounds,
      MissingEvalF,
      MissingEvalGradF,
  };

  /** Function for creating a new Ipopt Problem object.  This function
   *  returns an object that can be passed to the IpoptSolve call.  It
   *  contains the basic definition of the optimization problem, such
   *  as number of variables and constraints, bounds on variables and
   *  constraints, information about the derivatives, and the callback
   *  function for the computation of the optimization problem
   *  functions and derivatives.  During this call, the options file
   *  PARAMS.DAT is read as well.
   *
   *  If NULL is returned, there was a problem with one of the inputs
   *  or reading the options file. */
  IPOPT_EXPORT(enum CreateProblemStatus) CreateIpoptProblem(
      IpoptProblem * const p /** Output problem */
    , Index index_style   /** indexing style for iRow & jCol,
				 0 for C style, 1 for Fortran style */
    , Sizes_CB sizes      /** Callback function for setting sizes of
                              arrays that store variables, constraint
                              values and derivatives. */
    , Init_CB init        /** Callback function for initializing
                              variables and multipliers.  */
    , Bounds_CB bounds    /** Callback function for setting lower and
                              upper bounds on variable and constraints.
                              */
    , Eval_F_CB eval_f    /** Callback function for evaluating
                              objective function */
    , Eval_G_CB eval_g    /** Callback function for evaluating
                              constraint functions */
    , Eval_Grad_F_CB eval_grad_f
                          /** Callback function for evaluating gradient
                              of objective function */
    , Eval_Jac_G_CB eval_jac_g
                          /** Callback function for evaluating Jacobian
                              of constraint functions */
    , Eval_H_CB eval_h    /** Callback function for evaluating Hessian
                              of Lagrangian function */
    , ScalingParams_CB scaling/** Callback function for setting scaling
                                This function pointer can be Null */
  );

  /** Method for freeing a previously created IpoptProblem.  After
      freeing an IpoptProblem, it cannot be used anymore. */
  IPOPT_EXPORT(void) FreeIpoptProblem(IpoptProblem ipopt_problem);


  /** Function for adding a string option.  Returns FALSE the option
   *  could not be set (e.g., if keyword is unknown) */
  IPOPT_EXPORT(Bool) AddIpoptStrOption(IpoptProblem ipopt_problem, const char* keyword, const char* val);

  /** Function for adding a Number option.  Returns FALSE the option
   *  could not be set (e.g., if keyword is unknown) */
  IPOPT_EXPORT(Bool) AddIpoptNumOption(IpoptProblem ipopt_problem, const char* keyword, Number val);

  /** Function for adding an Int option.  Returns FALSE the option
   *  could not be set (e.g., if keyword is unknown) */
  IPOPT_EXPORT(Bool) AddIpoptIntOption(IpoptProblem ipopt_problem, const char* keyword, Int val);

  /** Function for opening an output file for a given name with given
   *  printlevel.  Returns false, if there was a problem opening the
   *  file. */
  IPOPT_EXPORT(Bool) OpenIpoptOutputFile(IpoptProblem ipopt_problem, const char* file_name,
                           Int print_level);

  /** Setting a callback function for the "intermediate callback"
   *  method in the TNLP.  This gives control back to the user once
   *  per iteration.  If set, it provides the user with some
   *  information on the state of the optimization.  This can be used
   *  to print some user-defined output.  It also gives the user a way
   *  to terminate the optimization prematurely.  If the callback
   *  method returns false, Ipopt will terminate the optimization.
   *  Calling this set method to set the CB pointer to NULL disables
   *  the intermediate callback functionality. */
  IPOPT_EXPORT(Bool) SetIntermediateCallback(IpoptProblem ipopt_problem,
					     Intermediate_CB intermediate_cb);

  struct SolverData {
      Number* x;         /** Optimal solution */
      Number* mult_g;    /** Final multipliers for constraints */
      Number* mult_x_L;  /** Final multipliers for lower variable bounds */
      Number* mult_x_U;  /** Final multipliers for upper variable */
  };

  struct SolveResult {
      struct SolverData data;
      Number  obj_val;   /** Final value of objective function */
      const Number* g;   /** Values of constraint at final point */
      enum ApplicationReturnStatus status; /** Return status */
  };

  /** Function calling the Ipopt optimization algorithm for a problem
      previously defined with CreateIpoptProblem.  The return
      specified outcome of the optimization procedure (e.g., success,
      failure etc).
   */
  IPOPT_EXPORT(struct SolveResult) IpoptSolve(
      IpoptProblem ipopt_problem
                         /** Problem that is to be optimized.  Ipopt
                             will use the options previously specified with
                             AddIpoptOption (etc) for this problem. */
    , UserDataPtr user_data
                         /** Pointer to user data.  This will be
                             passed unmodified to the callback
                             functions. */
  );

  /**
   * Retrieve solver data which can be modified between solves for warm starts.
   */
  IPOPT_EXPORT(struct SolverData) GetSolverData(
      IpoptProblem ipopt_problem
  );

#ifdef __cplusplus
} /* extern "C" { */
#endif

#undef IPOPT_EXPORT

#endif
