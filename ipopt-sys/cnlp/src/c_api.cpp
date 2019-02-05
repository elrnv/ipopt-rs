#include "c_api.h"
#include "nlp.hpp"

#include <coin/IpIpoptApplication.hpp>
#include <memory>
#include <vector>
#include <iterator>

using namespace std;

enum CNLP_CreateProblemStatus cnlp_create_problem(
        CNLP_ProblemPtr * const out_problem,
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
{
    // make sure input is Ok
    if ( !init ) {
        return CNLP_MISSING_INITIAL_GUESS;
    } else if ( !sizes ) {
        return CNLP_MISSING_SIZES;
    } else if ( !bounds ) {
        return CNLP_MISSING_BOUNDS;
    } else if ( !eval_f ) {
        return CNLP_MISSING_EVAL_F;
    } else if ( !eval_grad_f ) {
        return CNLP_MISSING_EVAL_GRAD_F;
    }

    CNLP_ProblemPtr problem = nullptr;

    Ipopt::IpoptApplication *app = new Ipopt::IpoptApplication;

    Ipopt::ApplicationReturnStatus status;

    try {
        // Create the original nlp
        problem = new CNLP_Problem(
                app,
                index_style,
                sizes,
                init,
                bounds,
                eval_f,
                eval_g,
                eval_grad_f,
                eval_jac_g,
                eval_h,
                scaling);
    }
    catch (INVALID_NLP& exc) {
        exc.ReportException(*app->Jnlst(), Ipopt::J_ERROR);
        status = Ipopt::Invalid_Problem_Definition;
    }
    catch( Ipopt::IpoptException& exc ) {
        exc.ReportException(*app->Jnlst(), Ipopt::J_ERROR);
        status = Ipopt::Unrecoverable_Exception;
    }

    app->RethrowNonIpoptException(false);

    *out_problem = problem;
    return CNLP_SUCCESS;
}

void cnlp_free_problem(CNLP_ProblemPtr problem)
{
    delete problem;
}

CNLP_Bool cnlp_add_str_option(CNLP_ProblemPtr problem, const char* keyword, const char* val)
{
    std::string tag(keyword);
    std::string value(val);
    return (CNLP_Bool) problem->get_app()->Options()->SetStringValue(tag, value);
}

CNLP_Bool cnlp_add_num_option(CNLP_ProblemPtr problem, const char* keyword, CNLP_Number val)
{
    std::string tag(keyword);
    Ipopt::Number value = val;
    return (CNLP_Bool) problem->get_app()->Options()->SetNumericValue(tag, value);
}

CNLP_Bool cnlp_add_int_option(CNLP_ProblemPtr problem, const char* keyword, CNLP_Int val)
{
    std::string tag(keyword);
    Ipopt::Index value = val;
    return (CNLP_Bool) problem->get_app()->Options()->SetIntegerValue(tag, value);
}

CNLP_Bool cnlp_open_output_file(CNLP_ProblemPtr problem, const char* file_name,
                                CNLP_Int print_level)
{
    std::string name(file_name);
    Ipopt::EJournalLevel level = Ipopt::EJournalLevel(print_level);
    return (CNLP_Bool) problem->get_app()->OpenOutputFile(name, level);
}

void cnlp_set_intermediate_callback(CNLP_ProblemPtr problem,
        CNLP_Intermediate_CB intermediate_cb)
{
    problem->set_intermediate_cb(intermediate_cb);
}

CNLP_SolveResult cnlp_solve(CNLP_ProblemPtr problem, CNLP_UserDataPtr user_data)
{
    return problem->solve(user_data);
}

CNLP_Bool cnlp_init_solution(CNLP_ProblemPtr problem, CNLP_UserDataPtr user_data)
{
    problem->set_user_data(user_data);
    return (CNLP_Bool) problem->init_solution();
}

CNLP_SolverData cnlp_get_solver_data(CNLP_ProblemPtr problem)
{
    return problem->get_solution_arguments();
}

