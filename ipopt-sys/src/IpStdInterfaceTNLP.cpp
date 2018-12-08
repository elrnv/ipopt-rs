// Copyright (C) 2004, 2010 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpStdInterfaceTNLP.cpp 2462 2014-02-01 04:17:44Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-09-02
// Edited by: Egor Larionov (2018)
//
#include <algorithm>
#include "IpStdInterfaceTNLP.hpp"
#include "IpBlas.hpp"

namespace Ipopt
{
  StdInterfaceTNLP::StdInterfaceTNLP(Index index_style,
                                     Sizes_CB sizes,
                                     Init_CB init,
                                     Bounds_CB bounds,
                                     Eval_F_CB eval_f,
                                     Eval_G_CB eval_g,
                                     Eval_Grad_F_CB eval_grad_f,
                                     Eval_Jac_G_CB eval_jac_g,
                                     Eval_H_CB eval_h,
                                     ScalingParams_CB scaling,
                                     Intermediate_CB* intermediate_cb)
      : TNLP()
      , index_style_(index_style)
      , sizes_(sizes)
      , init_(init)
      , bounds_(bounds)
      , eval_f_(eval_f)
      , eval_g_(eval_g)
      , eval_grad_f_(eval_grad_f)
      , eval_jac_g_(eval_jac_g)
      , eval_h_(eval_h)
      , scaling_(scaling)
      , intermediate_cb_(intermediate_cb)
      , user_data_(nullptr)
      , obj_sol_(0.0)
  {
    ASSERT_EXCEPTION(index_style_ == 0 || index_style_ == 1, INVALID_STDINTERFACE_NLP,
                     "Valid index styles are 0 (C style) or 1 (Fortran style)");


    ASSERT_EXCEPTION(sizes_, INVALID_STDINTERFACE_NLP,
                     "No callback for settings sizes of variable and derivative arrays provided.");
    ASSERT_EXCEPTION(init_, INVALID_STDINTERFACE_NLP,
                     "No callback for initializing variable and multipliers provided.");
    ASSERT_EXCEPTION(bounds_, INVALID_STDINTERFACE_NLP,
                     "No callback for setting bounds on variables and constraints provided.");
    ASSERT_EXCEPTION(eval_f_, INVALID_STDINTERFACE_NLP,
                     "No callback function for evaluating the value of objective function provided.");
    ASSERT_EXCEPTION(eval_g_, INVALID_STDINTERFACE_NLP,
                     "No callback function for evaluating the values of constraints provided.");
    ASSERT_EXCEPTION(eval_grad_f_, INVALID_STDINTERFACE_NLP,
                     "No callback function for evaluating the gradient of objective function provided.");
    ASSERT_EXCEPTION(eval_jac_g_, INVALID_STDINTERFACE_NLP,
                     "No callback function for evaluating the Jacobian of the constraints provided.");
    ASSERT_EXCEPTION(eval_h_, INVALID_STDINTERFACE_NLP,
                     "No callback function for evaluating the Hessian of the constraints provided.");
  }

  SolverData StdInterfaceTNLP::get_solution_arguments() {
    SolverData data;
    data.x = x_sol_.data();
    data.mult_g = lambda_sol_.data();
    data.mult_x_L = z_L_sol_.data();
    data.mult_x_U = z_U_sol_.data();
    return data;
  }

  Number StdInterfaceTNLP::get_objective_value() {
    return obj_sol_;
  }

  Number* StdInterfaceTNLP::get_constraint_function_values() {
    return g_sol_.data();
  }

  StdInterfaceTNLP::~StdInterfaceTNLP() { }

  void StdInterfaceTNLP::set_user_data(UserDataPtr user_data) {
      this->user_data_ = user_data;
  }

  bool StdInterfaceTNLP::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                                      Index& nnz_h_lag, IndexStyleEnum& index_style)
  {
    Bool retval = (*sizes_)(&n, &m, &nnz_jac_g, &nnz_h_lag, user_data_);

    x_sol_.resize(static_cast<std::size_t>(n), 0.0);
    z_L_sol_.resize(static_cast<std::size_t>(n), 0.0);
    z_U_sol_.resize(static_cast<std::size_t>(n), 0.0);
    g_sol_.resize(static_cast<std::size_t>(m), 0.0);
    lambda_sol_.resize(static_cast<std::size_t>(m), 0.0);

    index_style = (index_style_ == 0) ? C_STYLE : FORTRAN_STYLE;

    return (retval != 0);
  }

  bool StdInterfaceTNLP::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                         Index m, Number* g_l, Number* g_u)
  {
    Bool retval = (*bounds_)(n, x_l, x_u, m, g_l, g_u, user_data_);
    return (retval!=0);
  }

  bool StdInterfaceTNLP::get_scaling_parameters(
    Number& obj_scaling,
    bool& use_x_scaling, Index n,
    Number* x_scaling,
    bool& use_g_scaling, Index m,
    Number* g_scaling)
  {
    // If the user didn't provide a scaling function, just disable scaling.
    if (!scaling_) {
      obj_scaling = 1.0;
      use_x_scaling = false;
      use_g_scaling = false;
      return true;
    }

    // Otherwise call the user provided callback.

    Bool use_x = 0;
    Bool use_g = 0;
    Bool retval = (*scaling_)(&obj_scaling, &use_x, n, x_scaling, &use_g, m, g_scaling, user_data_);
    if (retval != 0) {
      use_x_scaling = use_x != 0;
      use_g_scaling = use_g != 0;
      return true;
    }

    return false;
  }

  bool StdInterfaceTNLP::get_starting_point(Index n, bool init_x,
      Number* x, bool init_z,
      Number* z_L, Number* z_U,
      Index m, bool init_lambda,
      Number* lambda)
  {
    Bool retval = (*init_)(n, (Bool)init_x, x, (Bool)init_z, z_L, z_U, m, (Bool)init_lambda, lambda, user_data_);
    return (retval!=0);
  }

  bool StdInterfaceTNLP::eval_f(Index n, const Number* x, bool new_x,
                                Number& obj_value)
  {
    Bool retval = (*eval_f_)(n, x, (Bool)new_x, &obj_value, user_data_);
    return (retval!=0);
  }

  bool StdInterfaceTNLP::eval_grad_f(Index n, const Number* x, bool new_x,
                                     Number* grad_f)
  {
    Bool retval = (*eval_grad_f_)(n, x, (Bool)new_x, grad_f, user_data_);
    return (retval!=0);
  }

  bool StdInterfaceTNLP::eval_g(Index n, const Number* x, bool new_x,
                                Index m, Number* g)
  {
    Bool retval = (*eval_g_)(n, x, (Bool)new_x, m, g, user_data_);
    return (retval!=0);
  }

  bool StdInterfaceTNLP::eval_jac_g(Index n, const Number* x, bool new_x,
                                    Index m, Index nele_jac, Index* iRow,
                                    Index *jCol, Number* values)
  {
    Bool retval=1;

    if ( (iRow && jCol && !values) || (!iRow && !jCol && values) ) {
      retval = (*eval_jac_g_)(n, x, (Bool)new_x, m, nele_jac,
                              iRow, jCol, values, user_data_);
    }
    else {
      DBG_ASSERT(false && "Invalid combination of iRow, jCol, and values pointers");
    }
    return (retval!=0);
  }

  bool StdInterfaceTNLP::eval_h(Index n, const Number* x, bool new_x,
                                Number obj_factor, Index m,
                                const Number* lambda, bool new_lambda,
                                Index nele_hess, Index* iRow, Index* jCol,
                                Number* values)
  {
    Bool retval=1;

    if ( (iRow && jCol && !values) || (!iRow && !jCol && values) ) {
      retval = (*eval_h_)(n, x, (Bool)new_x, obj_factor, m,
                          lambda, (Bool)new_lambda, nele_hess,
                          iRow, jCol, values, user_data_);
    }
    else {
      DBG_ASSERT(false && "Invalid combination of iRow, jCol, and values pointers");
    }
    return (retval!=0);
  }

  bool StdInterfaceTNLP::intermediate_callback(AlgorithmMode mode,
      Index iter, Number obj_value,
      Number inf_pr, Number inf_du,
      Number mu, Number d_norm,
      Number regularization_size,
      Number alpha_du, Number alpha_pr,
      Index ls_trials,
      const IpoptData* ip_data,
      IpoptCalculatedQuantities* ip_cq)
  {
    Bool retval = 1;
    if (intermediate_cb_ && *intermediate_cb_) {
      retval = (**intermediate_cb_)((Index)mode, iter, obj_value, inf_pr, inf_du,
                                   mu, d_norm, regularization_size, alpha_du,
                                   alpha_pr, ls_trials, user_data_);
    }
    return (retval!=0);
  }

  void StdInterfaceTNLP::finalize_solution(SolverReturn status,
      Index n, const Number* x, const Number* z_L, const Number* z_U,
      Index m, const Number* g, const Number* lambda,
      Number obj_value,
      const IpoptData* ip_data,
      IpoptCalculatedQuantities* ip_cq)
  {
    DBG_ASSERT(x_sol_.size() == n);
    DBG_ASSERT(z_L_sol_.size() == n);
    DBG_ASSERT(z_U_sol_.size() == n);
    DBG_ASSERT(g_sol_.size() == m);
    DBG_ASSERT(lambda_sol_.size() == m);

    IpBlasDcopy(n, x, 1, x_sol_.data(), 1);
    IpBlasDcopy(n, z_L, 1, z_L_sol_.data(), 1);
    IpBlasDcopy(n, z_U, 1, z_U_sol_.data(), 1);
    IpBlasDcopy(m, g, 1, g_sol_.data(), 1);
    IpBlasDcopy(m, lambda, 1, lambda_sol_.data(), 1);
    obj_sol_ = obj_value;
    // don't need to store the status, we get the status from the OptimizeTNLP method
  }

} // namespace Ipopt



