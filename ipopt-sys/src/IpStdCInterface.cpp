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
  Intermediate_CB intermediate_cb;
  Ipopt::SmartPtr<Ipopt::IpoptApplication> app;
  Ipopt::SmartPtr<Ipopt::StdInterfaceTNLP> nlp;
  Index num_solves;
};

enum CreateProblemStatus CreateIpoptProblem(
  IpoptProblem * const out_problem,
  Index index_style,
  Sizes_CB sizes,
  Init_CB init,
  Bounds_CB bounds,
  Eval_F_CB eval_f,
  Eval_G_CB eval_g,
  Eval_Grad_F_CB eval_grad_f,
  Eval_Jac_G_CB eval_jac_g,
  Eval_H_CB eval_h,
  ScalingParams_CB scaling)
{
  using namespace Ipopt;

  // make sure input is Ok
  if ( !init ) {
      return MissingInitialGuess;
  } else if ( !sizes ) {
      return MissingSizes;
  } else if ( !bounds ) {
      return MissingBounds;
  } else if ( !eval_f ) {
      return MissingEvalF;
  } else if ( !eval_grad_f ) {
      return MissingEvalGradF;
  }

  IpoptProblem problem = new IpoptProblemInfo;

  problem->intermediate_cb = nullptr;

  problem->app = new IpoptApplication();

  problem->num_solves = 0;

  Ipopt::ApplicationReturnStatus status;
  try {
    // Create the original nlp
    problem->nlp = new StdInterfaceTNLP(
            index_style,
            sizes,
            init,
            bounds,
            eval_f,
            eval_g,
            eval_grad_f,
            eval_jac_g,
            eval_h,
            scaling,
            &problem->intermediate_cb);
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
  *out_problem = problem;
  return Success;
}

void FreeIpoptProblem(IpoptProblem ipopt_problem)
{
  ipopt_problem->nlp = NULL;
  ipopt_problem->app = NULL;
  delete ipopt_problem;
}

Bool AddIpoptStrOption(IpoptProblem ipopt_problem, const char* keyword, const char* val)
{
  std::string tag(keyword);
  std::string value(val);
  return (Bool) ipopt_problem->app->Options()->SetStringValue(tag, value);
}

Bool AddIpoptNumOption(IpoptProblem ipopt_problem, const char* keyword, Number val)
{
  std::string tag(keyword);
  Ipopt::Number value=val;
  return (Bool) ipopt_problem->app->Options()->SetNumericValue(tag, value);
}

Bool AddIpoptIntOption(IpoptProblem ipopt_problem, const char* keyword, Int val)
{
  std::string tag(keyword);
  Ipopt::Index value=val;
  return (Bool) ipopt_problem->app->Options()->SetIntegerValue(tag, value);
}

Bool OpenIpoptOutputFile(IpoptProblem ipopt_problem, const char* file_name,
                         Int print_level)
{
  std::string name(file_name);
  Ipopt::EJournalLevel level = Ipopt::EJournalLevel(print_level);
  return (Bool) ipopt_problem->app->OpenOutputFile(name, level);
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

  Ipopt::ApplicationReturnStatus status;

  try {
      if (ipopt_problem->num_solves == 0) {
          // Initialize and process options
          status = ipopt_problem->app->Initialize();
          if (status != Ipopt::Solve_Succeeded) {
            SolveResult res;
            res.data = GetSolverData(ipopt_problem);
            res.g = ipopt_problem->nlp->get_constraint_function_values();
            res.obj_val = ipopt_problem->nlp->get_objective_value();
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

  SolveResult res;
  res.data = GetSolverData(ipopt_problem);
  res.g = ipopt_problem->nlp->get_constraint_function_values();
  res.obj_val = ipopt_problem->nlp->get_objective_value();
  res.status = (::ApplicationReturnStatus) status;

  return res;
}

SolverData GetSolverData(IpoptProblem ipopt_problem)
{
  return ipopt_problem->nlp->get_solution_arguments();
}

