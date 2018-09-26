// Copyright (C) 2004, 2011 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpStdCInterface.cpp 2398 2013-10-19 18:08:59Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13
// Edited by: Egor Larionov (2018)

#include "IpStdCInterface.h"
#include "IpStdInterfaceTNLP.hpp"
#include "IpOptionsList.hpp"
#include "IpIpoptApplication.hpp"
#include <memory>
#include <vector>
#include <iterator>

using namespace std;

struct IpoptProblemInfo
{
  vector<Number> x_L;
  vector<Number> x_U;
  vector<Number> g_L;
  vector<Number> g_U;
  Index nele_jac;
  Index nele_hess;
  Index index_style;
  Eval_F_CB eval_f;
  Eval_G_CB eval_g;
  Eval_Grad_F_CB eval_grad_f;
  Eval_Jac_G_CB eval_jac_g;
  Eval_H_CB eval_h;
  Intermediate_CB intermediate_cb;
  Ipopt::SmartPtr<Ipopt::IpoptApplication> app;
  Number obj_scaling;
  vector<Number> x_scaling;
  vector<Number> g_scaling;
  Ipopt::SmartPtr<Ipopt::StdInterfaceTNLP> nlp;
  Index num_solves;
  vector<Number> x;
  vector<Number> g; // output only
  Number obj_val; // output only
  vector<Number> mult_g;
  vector<Number> mult_x_L;
  vector<Number> mult_x_U;
};

IpoptProblem CreateIpoptProblem(
  Index n,
  const Number* x_L,
  const Number* x_U,
  const Number* init_x,
  const Number* init_mult_x_L,
  const Number* init_mult_x_U,
  Index m,
  const Number* g_L,
  const Number* g_U,
  const Number* init_mult_g,
  Index nele_jac,
  Index nele_hess,
  Index index_style,
  Eval_F_CB eval_f,
  Eval_G_CB eval_g,
  Eval_Grad_F_CB eval_grad_f,
  Eval_Jac_G_CB eval_jac_g,
  Eval_H_CB eval_h)
{
  using namespace Ipopt;

  // make sure input is Ok
  if (n<1 || m<0 || !init_x || !x_L || !x_U || (m>0 && (!g_L || !g_U)) ||
      (m==0 && nele_jac != 0) || (m>0 && nele_jac < 1) || nele_hess < 0 ||
      !eval_f || !eval_grad_f || (m>0 && (!eval_g || !eval_jac_g))) {
    return NULL;
  }

  IpoptProblem problem = new IpoptProblemInfo;

  problem->x_L.reserve(n);
  copy_n(x_L, n, std::back_inserter(problem->x_L));
  problem->x_U.reserve(n);
  copy_n(x_U, n, std::back_inserter(problem->x_U));

  // Copy the starting point information
  problem->x.reserve(n);
  copy_n(init_x, n, back_inserter(problem->x));
  if (init_mult_x_L) {
    problem->mult_x_L.reserve(n);
    copy_n(init_mult_x_L, n, back_inserter(problem->mult_x_L));
  }
  if (init_mult_x_U) {
    problem->mult_x_U.reserve(n);
    copy_n(init_mult_x_U, n, back_inserter(problem->mult_x_U));
  }

  if (m>0) {
    problem->g.resize(m, 0); // initialize solution memory

    problem->g_L.reserve(m);
    copy_n(g_L, m, back_inserter(problem->g_L));
    problem->g_U.reserve(m);
    copy_n(g_U, m, back_inserter(problem->g_U));

    // Copy the starting point information
    if (init_mult_g) {
      problem->mult_g.reserve(m);
      copy_n(init_mult_g, m, back_inserter(problem->mult_g));
    }
  }

  problem->obj_val = 0.0;

  problem->nele_jac = nele_jac;
  problem->nele_hess = nele_hess;
  problem->index_style = index_style;
  problem->eval_f = eval_f;
  problem->eval_g = eval_g;
  problem->eval_grad_f = eval_grad_f;
  problem->eval_jac_g = eval_jac_g;
  problem->eval_h = eval_h;
  problem->intermediate_cb = nullptr;
  problem->x_scaling.reserve(n);
  problem->g_scaling.reserve(m);

  problem->app = new IpoptApplication();

  problem->obj_scaling = 1;

  problem->num_solves = 0;

  Ipopt::ApplicationReturnStatus status;
  try {
    // Create the original nlp
    problem->nlp = new StdInterfaceTNLP(
            n, problem->x_L.data(), problem->x_U.data(),
            m, problem->g_L.data(), problem->g_U.data(),
            problem->nele_jac,
            problem->nele_hess,
            problem->index_style,
            problem->x.data(),
            problem->mult_g.data(),
            problem->mult_x_L.data(),
            problem->mult_x_U.data(),
            problem->eval_f,
            problem->eval_g,
            problem->eval_grad_f,
            problem->eval_jac_g,
            problem->eval_h,
            &problem->intermediate_cb,
            problem->x.data(),
            problem->mult_x_L.data(),
            problem->mult_x_U.data(),
            problem->g.data(),
            problem->mult_g.data(), // outputs
            &problem->obj_val,
            nullptr,
            problem->obj_scaling,
            problem->x_scaling.data(),
            problem->g_scaling.data());
  }
  catch (INVALID_STDINTERFACE_NLP& exc) {
    exc.ReportException(*problem->app->Jnlst(), J_ERROR);
    status = Ipopt::Invalid_Problem_Definition;
  }
  catch( IpoptException& exc ) {
    exc.ReportException(*problem->app->Jnlst(), J_ERROR);
    status = Ipopt::Unrecoverable_Exception;
  }

  problem->app->RethrowNonIpoptException(false);

  return problem;
}

void FreeIpoptProblem(IpoptProblem ipopt_problem)
{
  ipopt_problem->nlp = NULL;
  ipopt_problem->app = NULL;
  delete ipopt_problem;
}


Bool AddIpoptStrOption(IpoptProblem ipopt_problem, char* keyword, char* val)
{
  std::string tag(keyword);
  std::string value(val);
  return (Bool) ipopt_problem->app->Options()->SetStringValue(tag, value);
}

Bool AddIpoptNumOption(IpoptProblem ipopt_problem, char* keyword, Number val)
{
  std::string tag(keyword);
  Ipopt::Number value=val;
  return (Bool) ipopt_problem->app->Options()->SetNumericValue(tag, value);
}

Bool AddIpoptIntOption(IpoptProblem ipopt_problem, char* keyword, Int val)
{
  std::string tag(keyword);
  Ipopt::Index value=val;
  return (Bool) ipopt_problem->app->Options()->SetIntegerValue(tag, value);
}

Bool OpenIpoptOutputFile(IpoptProblem ipopt_problem, char* file_name,
                         Int print_level)
{
  std::string name(file_name);
  Ipopt::EJournalLevel level = Ipopt::EJournalLevel(print_level);
  return (Bool) ipopt_problem->app->OpenOutputFile(name, level);
}

Bool SetIpoptProblemScaling(IpoptProblem ipopt_problem,
                            Number obj_scaling,
                            Number* x_scaling,
                            Number* g_scaling)
{
  ipopt_problem->obj_scaling = obj_scaling;
  if (x_scaling) {
    ipopt_problem->x_scaling.clear();
    copy_n(x_scaling, ipopt_problem->x.size(), back_inserter(ipopt_problem->x_scaling));
  }
  else {
    ipopt_problem->x_scaling.clear();
  }
  if (g_scaling) {
    ipopt_problem->g_scaling.clear();
    copy_n(g_scaling, ipopt_problem->g.size(), back_inserter(ipopt_problem->g_scaling));
  }
  else {
    ipopt_problem->g_scaling.clear();
  }

  return (Bool)true;
}

Bool SetIntermediateCallback(IpoptProblem ipopt_problem,
                             Intermediate_CB intermediate_cb)
{
  ipopt_problem->intermediate_cb = intermediate_cb;
  return (Bool)true;
}

SolveResult IpoptSolve(IpoptProblem ipopt_problem, UserDataPtr user_data)
{
  using namespace Ipopt;

  ipopt_problem->nlp->set_user_data(user_data);

  SmartPtr<TNLP> tnlp(ipopt_problem->nlp);

  SolveResult res;
  res.data = GetSolverData(ipopt_problem);
  res.g = ipopt_problem->g.data();
  res.obj_val = 0.0;

  Ipopt::ApplicationReturnStatus status;

  try {
      if (ipopt_problem->num_solves == 0) {
          // Initialize and process options
          status = ipopt_problem->app->Initialize();
          if (status != Ipopt::Solve_Succeeded) {
            res.status = (::ApplicationReturnStatus) status;
            return res;
          }
          // solve
          status = ipopt_problem->app->OptimizeTNLP(tnlp);
      } else {
          // re-solve
          status = ipopt_problem->app->ReOptimizeTNLP(tnlp);
      }
  }
  catch (INVALID_STDINTERFACE_NLP& exc) {
    exc.ReportException(*ipopt_problem->app->Jnlst(), J_ERROR);
    status = Ipopt::Invalid_Problem_Definition;
  }
  catch( IpoptException& exc ) {
    exc.ReportException(*ipopt_problem->app->Jnlst(), J_ERROR);
    status = Ipopt::Unrecoverable_Exception;
  }

  ipopt_problem->num_solves += 1;

  res.obj_val = ipopt_problem->obj_val;
  res.status = (::ApplicationReturnStatus) status;

  return res;
}

SolverData GetSolverData(IpoptProblem ipopt_problem)
{
  SolverData data;
  data.x = ipopt_problem->x.data();
  data.mult_g = ipopt_problem->mult_g.data();
  data.mult_x_L = ipopt_problem->mult_x_L.data();
  data.mult_x_U = ipopt_problem->mult_x_U.data();

  return data;
}

