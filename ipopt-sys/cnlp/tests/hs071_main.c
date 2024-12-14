#include <stdio.h>
#include <stddef.h>

#include "hs071_nlp.h"
#include "../src/c_api.h"

struct HS071Data {
    double g_offset_x;
    double g_offset_y;
};

int main(int argv, char* argc[])
{
    struct CNLP_Problem* nlp = NULL;
    cnlp_create_problem(
          &nlp,
          0, // C-style 0-based indexing
          hs071_sizes,
          hs071_init,
          hs071_bounds,
          hs071_eval_f,
          hs071_eval_g,
          hs071_eval_grad_f,
          hs071_eval_jac_g,
          hs071_eval_h,
          NULL
    );

    if (!nlp) {
        printf("Failed to create problem");
    }

    // Get the sizes from the callback
    CNLP_Index n, m, nnz_jac_g, nnz_h_lag;
    hs071_sizes(&n, &m, &nnz_jac_g, &nnz_h_lag, NULL);

    cnlp_add_int_option(nlp, "print_level", 5);
    cnlp_add_num_option(nlp, "tol", 1e-7);
    cnlp_add_str_option(nlp, "mu_strategy", "adaptive");
    cnlp_set_intermediate_callback(nlp, hs071_intermediate_callback);

    struct HS071Data data;
    data.g_offset_x = 0.0;
    data.g_offset_y = 0.0;

    struct CNLP_SolveResult sol = cnlp_solve(nlp, (CNLP_UserDataPtr) &data);

    if (sol.status != CNLP_USER_REQUESTED_STOP) {
        printf("FAILURE");
        return (int) sol.status;
    } else {
        printf("SUCCESS");
    }

    printf("\n\nSolution of the primal variables, x\n");
    for (CNLP_Index i=0; i<n; i++) {
        printf("x[%d] = %f\n", i, sol.data.x[i]);
    }

    printf("\n\nSolution of the bound multipliers, z_L and z_U\n");
    for (CNLP_Index i=0; i<n; i++) {
        printf("z_L[%d] = %e\n", i, sol.data.mult_x_L[i]);
    }
    for (CNLP_Index i=0; i<n; i++) {
        printf("z_U[%d] = %e\n", i, sol.data.mult_x_U[i]);
    }

    printf("\n\nObjective value\n");
    printf("f(x*) = %f\n", sol.obj_val);

    printf("\n\nFinal value of the constraints:\n");
    for (CNLP_Index i=0; i<m ;i++) {
        printf("g(%d) = %f\n", i, sol.g[i]);
    }

    // Now we are going to solve this problem again, but with slightly modified constraints.
    // We change the constraint offset of the first constraint a bit,
    // and resolve the problem using the warm start option.

    data.g_offset_x = 0.2;

    cnlp_add_str_option(nlp, "warm_start_init_point", "yes");
    cnlp_add_num_option(nlp, "bound_push", 1e-5);
    cnlp_add_num_option(nlp, "bound_frac", 1e-5);
    cnlp_set_intermediate_callback(nlp, NULL);

    struct CNLP_SolveResult sol2 = cnlp_solve(nlp, (CNLP_UserDataPtr) &data);

    if (sol2.status != CNLP_SOLVE_SUCCEEDED) {
        printf("FAILURE");
        return (int) sol2.status;
    } else {
        printf("SUCCESS");
    }

    printf("\n\nSolution of the primal variables, x\n");
    for (CNLP_Index i=0; i<n; i++) {
        printf("x[%d] = %f\n", i, sol2.data.x[i]);
    }

    printf("\n\nSolution of the bound multipliers, z_L and z_U\n");
    for (CNLP_Index i=0; i<n; i++) {
        printf("z_L[%d] = %e\n", i, sol2.data.mult_x_L[i]);
    }
    for (CNLP_Index i=0; i<n; i++) {
        printf("z_U[%d] = %e\n", i, sol2.data.mult_x_U[i]);
    }

    printf("\n\nObjective value\n");
    printf("f(x*) = %f\n", sol2.obj_val);

    printf("\n\nFinal value of the constraints:\n");
    for (CNLP_Index i=0; i<m ;i++) {
        printf("g(%d) = %f\n", i, sol2.g[i]);
    }

    /// The C interface requires freeing the problem manually.
    cnlp_free_problem(nlp);

    return (int) sol2.status;
}
