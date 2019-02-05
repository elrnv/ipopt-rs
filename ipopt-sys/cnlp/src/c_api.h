#ifndef __IPOPT_SYS_C_API_H__
#define __IPOPT_SYS_C_API_H__

#ifndef CNLP_API
#ifdef _MSC_VER
#define CNLP_API(type) __declspec(dllexport) type __cdecl
#else 
#define CNLP_API(type) type
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /** Return codes for the Optimize call for an application */
    enum CNLP_ApplicationReturnStatus
    {
        CNLP_SOLVE_SUCCEEDED=0,
        CNLP_SOLVED_TO_ACCEPTABLE_LEVEL=1,
        CNLP_INFEASIBLE_PROBLEM_DETECTED=2,
        CNLP_SEARCH_DIRECTION_BECOMES_TOO_SMALL=3,
        CNLP_DIVERGING_ITERATES=4,
        CNLP_USER_REQUESTED_STOP=5,
        CNLP_FEASIBLE_POINT_FOUND=6,

        CNLP_MAXIMUM_ITERATIONS_EXCEEDED=-1,
        CNLP_RESTORATION_FAILED=-2,
        CNLP_ERROR_IN_STEP_COMPUTATION=-3,
        CNLP_MAXIMUM_CPUTIME_EXCEEDED=-4,
        CNLP_NOT_ENOUGH_DEGREES_OF_FREEDOM=-10,
        CNLP_INVALID_PROBLEM_DEFINITION=-11,
        CNLP_INVALID_OPTION=-12,
        CNLP_INVALID_NUMBER_DETECTED=-13,

        CNLP_UNRECOVERABLE_EXCEPTION=-100,
        CNLP_NONIPOPT_EXCEPTION_THROWN=-101,
        CNLP_INSUFFICIENT_MEMORY=-102,
        CNLP_INTERNAL_ERROR=-199
    };

    /** An internal state of the Ipopt algorithm. This is reported in intermediate callbacks. */
    enum CNLP_AlgorithmMode
    {
        CNLP_REGULAR_MODE = 0,
        CNLP_RESTORATION_PHASE_MODE = 1
    };

    /** The following typedefs must match the typedefs in Ipopt */
    //@{
    typedef double CNLP_Number;
    typedef int CNLP_Index;
    typedef int CNLP_Int;
    //@}

    /**
     * Structure collecting all information about the problem definition and solve statistics etc.
     * This is defined in the source file.
     */
    struct CNLP_Problem;

    /** Pointer to a CNLP_Problem */
    typedef struct CNLP_Problem* CNLP_ProblemPtr;

    /** Define a boolean type for C */
    typedef int CNLP_Bool;

    /**
     * A pointer for anything that is to be passed between the called and individual callback
     * function.
     */
    typedef void * CNLP_UserDataPtr;

    /**
     * Type defining the callback function for setting scaling parameters. This method is called if
     * "nlp_scaling_method" is set to "user-scaling". This function is optional.
     */
    typedef CNLP_Bool (*CNLP_ScalingParams_CB)(
            CNLP_Number* obj_scaling,
            CNLP_Bool* use_x_scaling, CNLP_Index n,
            CNLP_Number* x_scaling,
            CNLP_Bool* use_g_scaling, CNLP_Index m,
            CNLP_Number* g_scaling,
            CNLP_UserDataPtr user_data);

    /**
     * Type defining the callback function for setting sizes for arrays that will store variables,
     * constraint values and derivatives.
     */
    typedef CNLP_Bool (*CNLP_Sizes_CB)(
            CNLP_Index *n, CNLP_Index *m,
            CNLP_Index *nnz_jac_g, CNLP_Index *nnz_h_lag,
            CNLP_UserDataPtr user_data);

    /**
     * Type defining the callback function for initializing variables and multipliers.
     */
    typedef CNLP_Bool (*CNLP_Init_CB)(
            CNLP_Index n, CNLP_Bool init_x, CNLP_Number* x, 
            CNLP_Bool init_z, CNLP_Number* z_L, CNLP_Number* z_U, 
            CNLP_Index m, CNLP_Bool init_lambda, CNLP_Number* lambda, 
            CNLP_UserDataPtr user_data);

    /**
     * Type defining the callback function for specifying variable and constraint lower and upper
     * bounds.
     */
    typedef CNLP_Bool (*CNLP_Bounds_CB)(
            CNLP_Index n, CNLP_Number* x_l, CNLP_Number* x_u,
            CNLP_Index m, CNLP_Number* g_l, CNLP_Number* g_u,
            CNLP_UserDataPtr user_data);

    /**
     * Type defining the callback function for evaluating the value of the objective function.
     * Return value should be set to 0 if there was a problem doing the evaluation.
     */
    typedef CNLP_Bool (*CNLP_Eval_F_CB)(
            CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x,
            CNLP_Number* obj_value, CNLP_UserDataPtr user_data);

    /**
     * Type defining the callback function for evaluating the gradient of the objective function.
     * Return value should be set to 0 if there was a problem doing the evaluation.
     */
    typedef CNLP_Bool (*CNLP_Eval_Grad_F_CB)(
            CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x,
            CNLP_Number* grad_f, CNLP_UserDataPtr user_data);

    /**
     * Type defining the callback function for evaluating the value of the constraint functions.
     * Return value should be set to 0 if there was a problem doing the evaluation.
     */
    typedef CNLP_Bool (*CNLP_Eval_G_CB)(
            CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x,
            CNLP_Index m, CNLP_Number* g, CNLP_UserDataPtr user_data);

    /**
     * Type defining the callback function for evaluating the Jacobian of the constrant functions.
     * Return value should be set to 0 if there was a problem doing the evaluation.
     */
    typedef CNLP_Bool (*CNLP_Eval_Jac_G_CB)(
            CNLP_Index n, const CNLP_Number *x, CNLP_Bool new_x,
            CNLP_Index m, CNLP_Index nele_jac,
            CNLP_Index *iRow, CNLP_Index *jCol, CNLP_Number *values,
            CNLP_UserDataPtr user_data);

    /**
     * Type defining the callback function for evaluating the Hessian of the Lagrangian function.
     * Return value should be set to 0 if there was a problem doing the evaluation.
     */
    typedef CNLP_Bool (*CNLP_Eval_H_CB)(
            CNLP_Index n, const CNLP_Number *x, CNLP_Bool new_x, CNLP_Number obj_factor,
            CNLP_Index m, const CNLP_Number *lambda, CNLP_Bool new_lambda,
            CNLP_Index nele_hess, CNLP_Index *iRow, CNLP_Index *jCol,
            CNLP_Number *values, CNLP_UserDataPtr user_data);

    /**
     * Type defining the callback function for giving intermediate execution control to the user.
     * If set, it is called once per iteration, providing the user with some information on the
     * state of the optimization.  This can be used to print some user-defined output.  It also
     * gives the user a way to terminate the optimization prematurely.  If this method returns
     * false, Ipopt will terminate the optimization.
     */
    typedef CNLP_Bool (*CNLP_Intermediate_CB)(
            enum CNLP_AlgorithmMode alg_mod,
            CNLP_Index iter_count, CNLP_Number obj_value,
            CNLP_Number inf_pr, CNLP_Number inf_du,
            CNLP_Number mu, CNLP_Number d_norm,
            CNLP_Number regularization_size,
            CNLP_Number alpha_du, CNLP_Number alpha_pr,
            CNLP_Index ls_trials, CNLP_UserDataPtr user_data);

    /** Enum reporting the status of problem creation */
    enum CNLP_CreateProblemStatus {
        CNLP_SUCCESS,
        CNLP_MISSING_SIZES,
        CNLP_MISSING_INITIAL_GUESS,
        CNLP_MISSING_BOUNDS,
        CNLP_MISSING_EVAL_F,
        CNLP_MISSING_EVAL_GRAD_F,
        CNLP_INVALID_PROBLEM_DEFINITION_ON_CREATE,
        CNLP_UNRECOVERABLE_EXCEPTION_ON_CREATE,
    };

    /**
     * Function for creating a new CNLP_Problem object.  This function returns an object that can
     * be passed to the cnlp_solve call.  It contains the basic definition of the optimization
     * problem via various callbacks.
     */
    CNLP_API(enum CNLP_CreateProblemStatus) cnlp_create_problem(
            CNLP_ProblemPtr * const p // Output problem
            , CNLP_Index index_style  // indexing style for iRow & jCol, 0 for C style, 1 for
                                      // Fortran style.
            , CNLP_Sizes_CB sizes     // Callback function for setting sizes of
                                      // arrays that store variables, constraint values and
                                      // derivatives.
            , CNLP_Init_CB init       // Callback function for initializing variables and
                                      // multipliers.
            , CNLP_Bounds_CB bounds   // Callback function for setting lower and upper bounds on
                                      // variable and constraints.
            , CNLP_Eval_F_CB eval_f   // Callback function for evaluating objective function.
            , CNLP_Eval_G_CB eval_g   // Callback function for evaluating constraint functions.
            , CNLP_Eval_Grad_F_CB eval_grad_f // Callback function for evaluating gradient of
                                              // objective function
            , CNLP_Eval_Jac_G_CB eval_jac_g // Callback function for evaluating Jacobian of
                                            // constraint functions
            , CNLP_Eval_H_CB eval_h         // Callback function for evaluating Hessian of
                                            // Lagrangian function
            , CNLP_ScalingParams_CB scaling // Callback function for setting scaling This function
                                            // pointer can be Null
            );

    /**
     * Method for freeing a previously created CNLP_Problem. After freeing an CNLP_Problem, it
     * cannot be used anymore.
     */
    CNLP_API(void) cnlp_free_problem(CNLP_ProblemPtr problem);

    /**
     * Function for adding a string option. Returns 0 if the option could not be set (e.g., if the
     * keyword is unknown)
     */
    CNLP_API(CNLP_Bool) cnlp_add_str_option(CNLP_ProblemPtr problem, const char* keyword,
                                            const char* val);

    /**
     * Function for adding a CNLP_Number option. Returns 0 if the option could not be set (e.g., if
     * the keyword is unknown)
     */
    CNLP_API(CNLP_Bool) cnlp_add_num_option(CNLP_ProblemPtr problem, const char* keyword,
                                            CNLP_Number val);

    /**
     * Function for adding a CNLP_Int option. Returns 0 if the option could not be set (e.g., if
     * the keyword is unknown)
     */
    CNLP_API(CNLP_Bool) cnlp_add_int_option(CNLP_ProblemPtr problem, const char* keyword,
                                            CNLP_Int val);

    /**
     * Function for opening an output file for a given name with given printlevel.  Returns 0
     * if there was a problem opening the file.
     */
    CNLP_API(CNLP_Bool) cnlp_open_output_file(CNLP_ProblemPtr problem, const char* file_name,
                                              CNLP_Int print_level);

    /**
     * Setting a callback function for the "intermediate callback" method in the TNLP.  This gives
     * control back to the user once per iteration.  If set, it provides the user with some
     * information on the state of the optimization.  This can be used to print some user-defined
     * output.  It also gives the user a way to terminate the optimization prematurely.  If the
     * callback method returns 0, Ipopt will terminate the optimization.  Calling this set method to
     * set the CB pointer to NULL disables the intermediate callback functionality.
     */
    CNLP_API(void) cnlp_set_intermediate_callback(CNLP_ProblemPtr problem,
                                                  CNLP_Intermediate_CB intermediate_cb);

    /** Solution data for one solve. */
    struct CNLP_SolverData {
        CNLP_Number* x;         // Optimal solution
        CNLP_Number* mult_g;    // Final multipliers for constraints
        CNLP_Number* mult_x_L;  // Final multipliers for lower variable bounds
        CNLP_Number* mult_x_U;  // Final multipliers for upper variable
    };

    /** The result of one solve including solution and end state */
    struct CNLP_SolveResult {
        struct CNLP_SolverData data; // Solution data
        CNLP_Number  obj_val;        // Final value of objective function
        const CNLP_Number* g;        // Values of constraint at final poin
        enum CNLP_ApplicationReturnStatus status; // Return status
    };

    /**
     * Function calling the Ipopt optimization algorithm for a problem previously defined with
     * cnlp_create_problem.  The return specified outcome of the optimization procedure (e.g.,
     * success, failure etc).
     */
    CNLP_API(struct CNLP_SolveResult) cnlp_solve(
            CNLP_ProblemPtr problem // Problem that is to be optimized.  Ipopt
                                    // will use the options previously specified with
                                    // cnlp_add_*_option (etc) for this problem.
            , CNLP_UserDataPtr user_da // Pointer to user data.  This will be
                                       // passed unmodified to the callback functions.
            );

    /**
     * Initialize the solution vectors in the nlp. Calling this is required before calling
     * cnlp_get_solver_data. This function will call the necessary initialization callbacks provided
     * by the user.
     */
    CNLP_API(CNLP_Bool) cnlp_init_solution(CNLP_ProblemPtr problem,
                                          CNLP_UserDataPtr user_data);

    /**
     * Retrieve solver data for review without having to keep the result of cnlp_solve around.
     */
    CNLP_API(struct CNLP_SolverData) cnlp_get_solver_data(CNLP_ProblemPtr problem);

#ifdef __cplusplus
} /* extern "C" { */
#endif

#undef CNLP_API
#endif
