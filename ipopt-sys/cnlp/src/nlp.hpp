#ifndef __NLP_HPP__
#define __NLP_HPP__

#include "c_api.h"

#include <coin/IpUtils.hpp>
#include <coin/IpTNLP.hpp>
#include <coin/IpException.hpp>
#include <coin/IpSmartPtr.hpp>
#include <coin/IpIpoptApplication.hpp>
#include <vector>

/** Declare excpetion that is thrown when invalid NLP data
*  is provided */
DECLARE_STD_EXCEPTION(INVALID_NLP);

struct CNLP_Problem : public Ipopt::TNLP
{
public:
    CNLP_Problem(
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
            CNLP_ScalingParams_CB scaling);

    // Get a raw pointer to the application object.
    Ipopt::IpoptApplication *get_app();

    /** Getters for the solution data */

    /**
     * Get optimal variables and multipliers. This function returns the arguments of the
     * optimization.
     */
    CNLP_SolverData get_solution_arguments();

    CNLP_Number get_objective_value();

    CNLP_Number* get_constraint_function_values();

    /** Default destructor */
    virtual ~CNLP_Problem();

    /// Allow the user to set the user data pointer after the problem has already been created.
    /// This must be set before calling any of the user specified callbacks.
    void set_user_data(CNLP_UserDataPtr user_data);

    // Set the intermediate callback.
    void set_intermediate_cb(CNLP_Intermediate_CB intermediate_cb);

    bool init_solution();

    void preallocate_solution_data(CNLP_Index n, CNLP_Index m);


    CNLP_SolveResult solve(CNLP_UserDataPtr user_data);

    /**@name methods to gather information about the NLP. These methods are
     * overloaded from TNLP. See TNLP for their more detailed documentation. */
    //@{
    /** returns dimensions of the nlp. Overloaded from TNLP */
    virtual bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                              Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style);

    /** returns bounds of the nlp. Overloaded from TNLP */
    virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                                 Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u);

    /** returns scaling parameters (if nlp_scaling_method is selected
     * as user-scaling). Overloaded from TNLP */
    virtual bool get_scaling_parameters(Ipopt::Number& obj_scaling,
                                        bool& use_x_scaling, Ipopt::Index n,
                                        Ipopt::Number* x_scaling,
                                        bool& use_g_scaling, Ipopt::Index m,
                                        Ipopt::Number* g_scaling);

    /** provides a starting point for the nlp variables. Overloaded from TNLP */
    virtual bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                                    bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                                    Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda);

    /** evaluates the objective value for the nlp. Overloaded from TNLP */
    virtual bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                        Ipopt::Number& obj_value);

    /** evaluates the gradient of the objective for the
     *  nlp. Overloaded from TNLP */
    virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                             Ipopt::Number* grad_f);

    /** evaluates the constraint residuals for the nlp. Overloaded from TNLP */
    virtual bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m,
                        Ipopt::Number* g);

    /** specifies the jacobian structure (if values is NULL) and
     *  evaluates the jacobian values (if values is not NULL) for the
     *  nlp. Overloaded from TNLP */
    virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m,
                            Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
                            Ipopt::Number* values);

    /** specifies the structure of the hessian of the lagrangian (if values is NULL) and
     *  evaluates the values (if values is not NULL). Overloaded from TNLP */
    virtual bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                        Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
                        bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
                        Ipopt::Index* jCol, Ipopt::Number* values);

    /** Intermediate Callback method for the user.  Overloaded from TNLP */
    virtual bool intermediate_callback(Ipopt::AlgorithmMode mode,
                                       Ipopt::Index iter, Ipopt::Number obj_value,
                                       Ipopt::Number inf_pr, Ipopt::Number inf_du,
                                       Ipopt::Number mu, Ipopt::Number d_norm,
                                       Ipopt::Number regularization_size,
                                       Ipopt::Number alpha_du, Ipopt::Number alpha_pr,
                                       Ipopt::Index ls_trials,
                                       const Ipopt::IpoptData* ip_data,
                                       Ipopt::IpoptCalculatedQuantities* ip_cq);
    //@}

    /** @name Solution Methods */
    //@{
    virtual void finalize_solution(Ipopt::SolverReturn status,
                                   Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U,
                                   Ipopt::Index m, const Ipopt::Number* g, const Ipopt::Number* lambda,
                                   Ipopt::Number obj_value,
                                   const Ipopt::IpoptData* ip_data,
                                   Ipopt::IpoptCalculatedQuantities* ip_cq);
    //@}
private:
    /** 
     * Helper function to build a solver result from a given status by collecting all the relevant
     * solve data into one struct.
     */
    CNLP_SolveResult build_solver_result(Ipopt::ApplicationReturnStatus status);

private:
    Ipopt::SmartPtr<Ipopt::IpoptApplication> m_app; // The application sets solver options
    std::size_t m_num_solves; // CNLP_Number of times the Solver ran for this instance

    const CNLP_Index m_index_style; // Starting value of the iRow and jCol parameters for matrices

    CNLP_Sizes_CB m_sizes; // Callback function setting array size information
    CNLP_Init_CB m_init; // Callback function initializing the iterates
    CNLP_Bounds_CB m_bounds; // Callback evaluating lower and upper bounds on variables and constraints
    CNLP_Eval_F_CB m_eval_f; // Callback function evaluating value of objective function.
    CNLP_Eval_G_CB m_eval_g; // Callback function evaluating value of constraints
    CNLP_Eval_Grad_F_CB m_eval_grad_f; // Callback function evaluating gradient of objective function
    CNLP_Eval_Jac_G_CB m_eval_jac_g; // Callback function evaluating Jacobian of constraints
    CNLP_Eval_H_CB m_eval_h; // Callback function evaluating Hessian of Lagrangian
    CNLP_ScalingParams_CB m_scaling; // Callback function for setting scaling parameters

    CNLP_Intermediate_CB m_intermediate_cb; // Intermediate callback function gives control to user
    CNLP_UserDataPtr m_user_data;

    /** Solution data */
    //@{
    std::vector<CNLP_Number> m_x_sol;
    std::vector<CNLP_Number> m_z_L_sol;
    std::vector<CNLP_Number> m_z_U_sol;
    std::vector<CNLP_Number> m_g_sol;
    std::vector<CNLP_Number> m_lambda_sol;
    CNLP_Number m_obj_sol;
    //@}

    /** Overloaded Equals Operator */
    void operator=(const CNLP_Problem&);

    /** Deleted Default Constructor */
    CNLP_Problem();

    /** Deleted Copy Constructor */
    CNLP_Problem(const CNLP_Problem&);

};

#endif
