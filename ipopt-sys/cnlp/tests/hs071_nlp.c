#include "hs071_nlp.h"

#include <assert.h>
#include <stddef.h>

CNLP_Bool hs071_sizes(CNLP_Index* n, CNLP_Index* m, CNLP_Index* nnz_jac_g,
                 CNLP_Index* nnz_h_lag, CNLP_UserDataPtr user_data)
{
  *n = 4; // 4 variables
  *m = 2; // 1 equality and 1 inequality constriant
  *nnz_jac_g = 8;
  *nnz_h_lag = 10;

  return 1;
}

CNLP_Bool hs071_bounds(CNLP_Index n, CNLP_Number* x_l, CNLP_Number* x_u,
                       CNLP_Index m, CNLP_Number* g_l, CNLP_Number* g_u,
                       CNLP_UserDataPtr user_data)
{
  assert(n == 4);
  assert(m == 2);
  for (CNLP_Index i=0; i<4; i++) {
    x_l[i] = 1.0;
  }

  for (CNLP_Index i=0; i<4; i++) {
    x_u[i] = 5.0;
  }

  g_l[0] = 25;
  g_u[0] = 2e19;

  g_l[1] = g_u[1] = 40.0;

  return 1;
}

CNLP_Bool hs071_init(CNLP_Index n, CNLP_Bool init_x, CNLP_Number* x,
                     CNLP_Bool init_z, CNLP_Number* z_L, CNLP_Number* z_U,
                     CNLP_Index m, CNLP_Bool init_lambda,
                     CNLP_Number* lambda,
                     CNLP_UserDataPtr user_data)
{
  assert(init_x == 1);
  x[0] = 1.0;
  x[1] = 5.0;
  x[2] = 5.0;
  x[3] = 1.0;

  if (init_z) {
      z_L[0] = 0.0;
      z_L[1] = 0.0;
      z_L[2] = 0.0;
      z_L[3] = 0.0;
      z_U[0] = 0.0;
      z_U[1] = 0.0;
      z_U[2] = 0.0;
      z_U[3] = 0.0;
  }

  if (init_lambda) {
      lambda[0] = 0.0;
      lambda[1] = 0.0;
  }

  return 1;
}

CNLP_Bool hs071_eval_f(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x, CNLP_Number* obj_value,
                       CNLP_UserDataPtr user_data)
{
  assert(n == 4);

  *obj_value = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];

  return 1;
}

// return the gradient of the objective function grad_{x} f(x)
CNLP_Bool hs071_eval_grad_f(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x, CNLP_Number* grad_f,
                            CNLP_UserDataPtr user_data)
{
  assert(n == 4);

  grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
  grad_f[1] = x[0] * x[3];
  grad_f[2] = x[0] * x[3] + 1;
  grad_f[3] = x[0] * (x[0] + x[1] + x[2]);

  return 1;
}

// return the value of the constraints: g(x)
CNLP_Bool hs071_eval_g(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x, CNLP_Index m, CNLP_Number* g, 
                       CNLP_UserDataPtr user_data)
{
  assert(n == 4);
  assert(m == 2);

  g[0] = x[0] * x[1] * x[2] * x[3];
  g[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];

  return 1;
}

// return the structure or values of the jacobian
CNLP_Bool hs071_eval_jac_g(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x,
                           CNLP_Index m, CNLP_Index nele_jac, CNLP_Index* iRow, CNLP_Index* jCol,
                           CNLP_Number* values, CNLP_UserDataPtr user_data)
{
  if (values == NULL) {
    iRow[0] = 0;
    jCol[0] = 0;
    iRow[1] = 0;
    jCol[1] = 1;
    iRow[2] = 0;
    jCol[2] = 2;
    iRow[3] = 0;
    jCol[3] = 3;
    iRow[4] = 1;
    jCol[4] = 0;
    iRow[5] = 1;
    jCol[5] = 1;
    iRow[6] = 1;
    jCol[6] = 2;
    iRow[7] = 1;
    jCol[7] = 3;
  } else {
    values[0] = x[1]*x[2]*x[3]; // 0,0
    values[1] = x[0]*x[2]*x[3]; // 0,1
    values[2] = x[0]*x[1]*x[3]; // 0,2
    values[3] = x[0]*x[1]*x[2]; // 0,3

    values[4] = 2*x[0]; // 1,0
    values[5] = 2*x[1]; // 1,1
    values[6] = 2*x[2]; // 1,2
    values[7] = 2*x[3]; // 1,3
  }

  return 1;
}

CNLP_Bool hs071_eval_h(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x,
                       CNLP_Number obj_factor, CNLP_Index m, const CNLP_Number* lambda,
                       CNLP_Bool new_lambda, CNLP_Index nele_hess, CNLP_Index* iRow,
                       CNLP_Index* jCol, CNLP_Number* values, CNLP_UserDataPtr user_data)
{
  if (values == NULL) {
    CNLP_Index idx=0;
    for (CNLP_Index row = 0; row < 4; row++) {
      for (CNLP_Index col = 0; col <= row; col++) {
        iRow[idx] = row;
        jCol[idx] = col;
        idx++;
      }
    }

    assert(idx == nele_hess);
  } else {
      // Objective
    values[0] = obj_factor * (2*x[3]); // 0,0

    values[1] = obj_factor * (x[3]);   // 1,0
    values[2] = 0.;                    // 1,1

    values[3] = obj_factor * (x[3]);   // 2,0
    values[4] = 0.;                    // 2,1
    values[5] = 0.;                    // 2,2

    values[6] = obj_factor * (2*x[0] + x[1] + x[2]); // 3,0
    values[7] = obj_factor * (x[0]);                 // 3,1
    values[8] = obj_factor * (x[0]);                 // 3,2
    values[9] = 0.;                                  // 3,3

    // Constraints
    values[1] += lambda[0] * (x[2] * x[3]); // 1,0

    values[3] += lambda[0] * (x[1] * x[3]); // 2,0
    values[4] += lambda[0] * (x[0] * x[3]); // 2,1

    values[6] += lambda[0] * (x[1] * x[2]); // 3,0
    values[7] += lambda[0] * (x[0] * x[2]); // 3,1
    values[8] += lambda[0] * (x[0] * x[1]); // 3,2

    values[0] += lambda[1] * 2; // 0,0

    values[2] += lambda[1] * 2; // 1,1

    values[5] += lambda[1] * 2; // 2,2

    values[9] += lambda[1] * 2; // 3,3
  }

  return 1;
}

CNLP_Bool hs071_intermediate_callback(enum CNLP_AlgorithmMode alg_mod,
                                 CNLP_Index iter_count,
                                 CNLP_Number obj_value,
                                 CNLP_Number inf_pr,
                                 CNLP_Number inf_du,
                                 CNLP_Number mu,
                                 CNLP_Number d_norm,
                                 CNLP_Number regularization_size,
                                 CNLP_Number alpha_du,
                                 CNLP_Number alpha_pr,
                                 CNLP_Index ls_trials,
                                 CNLP_UserDataPtr user_data)
{
  return inf_pr >= 1e-4;
}
