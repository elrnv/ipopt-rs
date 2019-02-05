#include "nlp.hpp"
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpBlas.hpp>

#include <algorithm>

/**
 * The following two functions provide safe conversion for return codes and modes in Ipopt.
 * Although not strictly necessary, this makes this interface a bit more robust against api changes
 * in Ipopt compared to naive casting.
 */
CNLP_ApplicationReturnStatus convert_application_status(Ipopt::ApplicationReturnStatus status)
{
    using namespace Ipopt;
    switch (status)
    {
        case Solve_Succeeded:                   return CNLP_SOLVE_SUCCEEDED;
        case Solved_To_Acceptable_Level:        return CNLP_SOLVED_TO_ACCEPTABLE_LEVEL;
        case Infeasible_Problem_Detected:       return CNLP_INFEASIBLE_PROBLEM_DETECTED;
        case Search_Direction_Becomes_Too_Small:return CNLP_SEARCH_DIRECTION_BECOMES_TOO_SMALL;
        case Diverging_Iterates:                return CNLP_DIVERGING_ITERATES;
        case User_Requested_Stop:               return CNLP_USER_REQUESTED_STOP;
        case Feasible_Point_Found:              return CNLP_FEASIBLE_POINT_FOUND;

        case Maximum_Iterations_Exceeded:       return CNLP_MAXIMUM_ITERATIONS_EXCEEDED;
        case Restoration_Failed:                return CNLP_RESTORATION_FAILED;
        case Error_In_Step_Computation:         return CNLP_ERROR_IN_STEP_COMPUTATION;
        case Maximum_CpuTime_Exceeded:          return CNLP_MAXIMUM_CPUTIME_EXCEEDED;
        case Not_Enough_Degrees_Of_Freedom:     return CNLP_NOT_ENOUGH_DEGREES_OF_FREEDOM;
        case Invalid_Problem_Definition:        return CNLP_INVALID_PROBLEM_DEFINITION;
        case Invalid_Option:                    return CNLP_INVALID_OPTION;
        case Invalid_Number_Detected:           return CNLP_INVALID_NUMBER_DETECTED;

        case Unrecoverable_Exception:           return CNLP_UNRECOVERABLE_EXCEPTION;
        case NonIpopt_Exception_Thrown:         return CNLP_NONIPOPT_EXCEPTION_THROWN;
        case Insufficient_Memory:               return CNLP_INSUFFICIENT_MEMORY;
        case Internal_Error:                    return CNLP_INTERNAL_ERROR;
    };
}

CNLP_AlgorithmMode convert_algorithm_mode(Ipopt::AlgorithmMode mode)
{
    switch (mode) {
        case Ipopt::RegularMode:          return CNLP_REGULAR_MODE;
        case Ipopt::RestorationPhaseMode: return CNLP_RESTORATION_PHASE_MODE;
    };
}

CNLP_Problem::CNLP_Problem(
        Ipopt::SmartPtr<Ipopt::IpoptApplication> app,
        CNLP_Index index_style,
        CNLP_Sizes_CB sizes,
        CNLP_Init_CB init,
        CNLP_Bounds_CB bounds,
        CNLP_Eval_F_CB eval_f,
        CNLP_Eval_G_CB eval_g,
        CNLP_Eval_Grad_F_CB eval_grad_f,
        CNLP_Eval_Jac_G_CB eval_jac_g,
        CNLP_Eval_H_CB eval_h,
        CNLP_ScalingParams_CB scaling)
    : TNLP()
    , m_app(app)
    , m_num_solves(0)
    , m_index_style(index_style)
    , m_sizes(sizes)
    , m_init(init)
    , m_bounds(bounds)
    , m_eval_f(eval_f)
    , m_eval_g(eval_g)
    , m_eval_grad_f(eval_grad_f)
    , m_eval_jac_g(eval_jac_g)
    , m_eval_h(eval_h)
    , m_scaling(scaling)
    , m_intermediate_cb(nullptr)
    , m_user_data(nullptr)
      , m_obj_sol(0.0)
{
    ASSERT_EXCEPTION(m_index_style == 0 || m_index_style == 1, INVALID_NLP,
            "Valid index styles are 0 (C style) or 1 (Fortran style)");
    ASSERT_EXCEPTION(m_sizes, INVALID_NLP,
            "No callback for settings sizes of variable and derivative arrays provided.");
    ASSERT_EXCEPTION(m_init, INVALID_NLP,
            "No callback for initializing variable and multipliers provided.");
    ASSERT_EXCEPTION(m_bounds, INVALID_NLP,
            "No callback for setting bounds on variables and constraints provided.");
    ASSERT_EXCEPTION(m_eval_f, INVALID_NLP,
            "No callback function for evaluating the value of objective function provided.");
    ASSERT_EXCEPTION(m_eval_g, INVALID_NLP,
            "No callback function for evaluating the values of constraints provided.");
    ASSERT_EXCEPTION(m_eval_grad_f, INVALID_NLP,
            "No callback function for evaluating the gradient of objective function provided.");
    ASSERT_EXCEPTION(m_eval_jac_g, INVALID_NLP,
            "No callback function for evaluating the Jacobian of the constraints provided.");
    ASSERT_EXCEPTION(m_eval_h, INVALID_NLP,
            "No callback function for evaluating the Hessian of the constraints provided.");
}

bool CNLP_Problem::init_solution() {
    DBG_ASSERT(m_user_data != nullptr && "User data has not been set prior to calling user specified callbacks.");

    // Preallocate solution arrays.
    CNLP_Index n, m;
    CNLP_Index nnz_jac_g, nnz_h_lag; // not used
    if ((*m_sizes)(&n, &m, &nnz_jac_g, &nnz_h_lag, m_user_data) == 0) {
        return false;
    }

    preallocate_solution_data(n, m);

    // Populate solution arrays with initial guess data.
    return get_starting_point(n, true, m_x_sol.data(), true, m_z_L_sol.data(), m_z_U_sol.data(), m, true, m_lambda_sol.data());
}

Ipopt::IpoptApplication *CNLP_Problem::get_app() {
    return Ipopt::GetRawPtr(m_app);
}

CNLP_SolverData CNLP_Problem::get_solution_arguments() {
    CNLP_SolverData data;
    data.x = m_x_sol.data();
    data.mult_g = m_lambda_sol.data();
    data.mult_x_L = m_z_L_sol.data();
    data.mult_x_U = m_z_U_sol.data();
    return data;
}

CNLP_Number CNLP_Problem::get_objective_value() {
    return m_obj_sol;
}

CNLP_Number* CNLP_Problem::get_constraint_function_values() {
    return m_g_sol.data();
}

CNLP_Problem::~CNLP_Problem() {
    m_app = nullptr;
}

void CNLP_Problem::set_user_data(CNLP_UserDataPtr user_data) {
    this->m_user_data = user_data;
}

void CNLP_Problem::set_intermediate_cb(CNLP_Intermediate_CB intermediate_cb) {
    this->m_intermediate_cb = intermediate_cb;
}

void CNLP_Problem::preallocate_solution_data(CNLP_Index n, CNLP_Index m) {
    m_x_sol.resize(static_cast<std::size_t>(n), 0.0);
    m_z_L_sol.resize(static_cast<std::size_t>(n), 0.0);
    m_z_U_sol.resize(static_cast<std::size_t>(n), 0.0);
    m_g_sol.resize(static_cast<std::size_t>(m), 0.0);
    m_lambda_sol.resize(static_cast<std::size_t>(m), 0.0);
}

CNLP_SolveResult CNLP_Problem::build_solver_result(Ipopt::ApplicationReturnStatus status) {
    CNLP_SolveResult res;
    res.data = get_solution_arguments();
    res.g = get_constraint_function_values();
    res.obj_val = get_objective_value();
    res.status = convert_application_status(status);
    return res;
}

CNLP_SolveResult CNLP_Problem::solve(CNLP_UserDataPtr user_data) {
    set_user_data(user_data);
    Ipopt::SmartPtr<TNLP> tnlp(this);
    this->AddRef(&tnlp); // Add an extra ref, since we don't want this deleted.
    Ipopt::ApplicationReturnStatus status;

    try {
        if (m_num_solves == 0) {
            // Initialize and process options
            status = m_app->Initialize();
            if (status != Ipopt::Solve_Succeeded) {
                return build_solver_result(status);
            }
            // Solve
            status = m_app->OptimizeTNLP(tnlp);
        } else {
            // Re-solve
            status = m_app->ReOptimizeTNLP(tnlp);
        }
    }
    catch (INVALID_NLP& e) {
        e.ReportException(*m_app->Jnlst(), Ipopt::J_ERROR);
        status = Ipopt::Invalid_Problem_Definition;
    }
    catch (Ipopt::IpoptException& e) {
        e.ReportException(*m_app->Jnlst(), Ipopt::J_ERROR);
        status = Ipopt::Unrecoverable_Exception;
    }

    m_num_solves += 1;

    return build_solver_result(status);
}

bool CNLP_Problem::get_nlp_info(
        Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
        Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style)
{
    CNLP_Bool retval = (*m_sizes)(&n, &m, &nnz_jac_g, &nnz_h_lag, m_user_data);
    preallocate_solution_data(n, m);

    index_style = (m_index_style == 0) ? C_STYLE : FORTRAN_STYLE;

    return (retval != 0);
}

bool CNLP_Problem::get_bounds_info(
        Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
        Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u)
{
    CNLP_Bool retval = (*m_bounds)(n, x_l, x_u, m, g_l, g_u, m_user_data);
    return (retval!=0);
}

bool CNLP_Problem::get_scaling_parameters(
        Ipopt::Number& obj_scaling,
        bool& use_x_scaling, Ipopt::Index n,
        Ipopt::Number* x_scaling,
        bool& use_g_scaling, Ipopt::Index m,
        Ipopt::Number* g_scaling)
{
    // If the user didn't provide a scaling function, just disable scaling.
    if (!m_scaling) {
        obj_scaling = 1.0;
        use_x_scaling = false;
        use_g_scaling = false;
        return true;
    }

    // Otherwise call the user provided callback.

    CNLP_Bool use_x = 0;
    CNLP_Bool use_g = 0;
    CNLP_Bool retval = (*m_scaling)(&obj_scaling, &use_x, n, x_scaling, &use_g, m, g_scaling, m_user_data);
    if (retval != 0) {
        use_x_scaling = use_x != 0;
        use_g_scaling = use_g != 0;
        return true;
    }

    return false;
}

bool CNLP_Problem::get_starting_point(
        Ipopt::Index n, bool init_x,
        Ipopt::Number* x, bool init_z,
        Ipopt::Number* z_L, Ipopt::Number* z_U,
        Ipopt::Index m, bool init_lambda,
        Ipopt::Number* lambda)
{
    CNLP_Bool retval = (*m_init)(n, (CNLP_Bool)init_x, x, (CNLP_Bool)init_z, z_L, z_U, m, (CNLP_Bool)init_lambda, lambda, m_user_data);
    return (retval!=0);
}

bool CNLP_Problem::eval_f(
        Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value)
{
    CNLP_Bool retval = (*m_eval_f)(n, x, (CNLP_Bool)new_x, &obj_value, m_user_data);
    return (retval!=0);
}

bool CNLP_Problem::eval_grad_f(
        Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f)
{
    CNLP_Bool retval = (*m_eval_grad_f)(n, x, (CNLP_Bool)new_x, grad_f, m_user_data);
    return (retval!=0);
}

bool CNLP_Problem::eval_g(
        Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g)
{
    CNLP_Bool retval = (*m_eval_g)(n, x, (CNLP_Bool)new_x, m, g, m_user_data);
    return (retval!=0);
}

bool CNLP_Problem::eval_jac_g(
        Ipopt::Index n, const Ipopt::Number* x, bool new_x,
        Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow,
        Ipopt::Index *jCol, Ipopt::Number* values)
{
    CNLP_Bool retval=1;

    if ( (iRow && jCol && !values) || (!iRow && !jCol && values) ) {
        retval = (*m_eval_jac_g)(n, x, (CNLP_Bool)new_x, m, nele_jac,
                iRow, jCol, values, m_user_data);
    }
    else {
        DBG_ASSERT(false && "Invalid combination of iRow, jCol, and values pointers");
    }
    return (retval!=0);
}

bool CNLP_Problem::eval_h(
        Ipopt::Index n, const Ipopt::Number* x, bool new_x,
        Ipopt::Number obj_factor, Ipopt::Index m,
        const Ipopt::Number* lambda, bool new_lambda,
        Ipopt::Index nele_hess, Ipopt::Index* iRow, CNLP_Index* jCol,
        Ipopt::Number* values)
{
    CNLP_Bool retval=1;

    if ( (iRow && jCol && !values) || (!iRow && !jCol && values) ) {
        retval = (*m_eval_h)(n, x, (CNLP_Bool)new_x, obj_factor, m,
                lambda, (CNLP_Bool)new_lambda, nele_hess,
                iRow, jCol, values, m_user_data);
    }
    else {
        DBG_ASSERT(false && "Invalid combination of iRow, jCol, and values pointers");
    }
    return (retval!=0);
}

bool CNLP_Problem::intermediate_callback(
        Ipopt::AlgorithmMode mode,
        Ipopt::Index iter, Ipopt::Number obj_value,
        Ipopt::Number inf_pr, Ipopt::Number inf_du,
        Ipopt::Number mu, Ipopt::Number d_norm,
        Ipopt::Number regularization_size,
        Ipopt::Number alpha_du, Ipopt::Number alpha_pr,
        Ipopt::Index ls_trials,
        const Ipopt::IpoptData* ip_data,
        Ipopt::IpoptCalculatedQuantities* ip_cq)
{
    CNLP_Bool retval = 1;
    if (m_intermediate_cb && *m_intermediate_cb) {
        retval = (**m_intermediate_cb)(convert_algorithm_mode(mode), iter, obj_value, inf_pr, inf_du,
                mu, d_norm, regularization_size, alpha_du,
                alpha_pr, ls_trials, m_user_data);
    }
    return (retval!=0);
}

void CNLP_Problem::finalize_solution(
        Ipopt::SolverReturn status,
        Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U,
        Ipopt::Index m, const Ipopt::Number* g, const Ipopt::Number* lambda,
        Ipopt::Number obj_value,
        const Ipopt::IpoptData* ip_data,
        Ipopt::IpoptCalculatedQuantities* ip_cq)
{
    DBG_ASSERT(m_x_sol.size() == n);
    DBG_ASSERT(m_z_L_sol.size() == n);
    DBG_ASSERT(m_z_U_sol.size() == n);
    DBG_ASSERT(m_g_sol.size() == m);
    DBG_ASSERT(m_lambda_sol.size() == m);

    Ipopt::IpBlasDcopy(n, x, 1, m_x_sol.data(), 1);
    Ipopt::IpBlasDcopy(n, z_L, 1, m_z_L_sol.data(), 1);
    Ipopt::IpBlasDcopy(n, z_U, 1, m_z_U_sol.data(), 1);
    Ipopt::IpBlasDcopy(m, g, 1, m_g_sol.data(), 1);
    Ipopt::IpBlasDcopy(m, lambda, 1, m_lambda_sol.data(), 1);
    m_obj_sol = obj_value;
    // don't need to store the status, we get the status from the OptimizeTNLP method
}

