//   Copyright 2018 Egor Larionov
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

use approx::{assert_relative_eq, relative_eq, __assert_approx};

use ipopt::*;

struct NLP {
    g_offset: [f64; 2],
    iterations: usize,
}
impl NLP {
    fn intermediate_cb(&mut self, data: IntermediateCallbackData) -> bool {
        self.count_iterations_cb(data);
        data.inf_pr >= 1e-4
    }
    fn count_iterations_cb(&mut self, data: IntermediateCallbackData) -> bool {
        self.iterations = data.iter_count as usize;
        true
    }
}

impl BasicProblem for NLP {
    fn num_variables(&self) -> usize {
        4
    }
    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        x_l.swap_with_slice(vec![1.0; 4].as_mut_slice());
        x_u.swap_with_slice(vec![5.0; 4].as_mut_slice());
        true
    }
    fn initial_point(&self, x: &mut [Number]) {
        x.swap_with_slice(vec![1.0, 5.0, 5.0, 1.0].as_mut_slice());
    }
    fn objective(&self, x: &[Number], obj: &mut Number) -> bool {
        *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        true
    }
    fn objective_grad(&self, x: &[Number], grad_f: &mut [Number]) -> bool {
        grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
        grad_f[1] = x[0] * x[3];
        grad_f[2] = x[0] * x[3] + 1.0;
        grad_f[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }

    // The following two callbacks are activated only when `nlp_scaling_method` is set to
    // `user-scaling`.
    fn variable_scaling(&self, x_scaling: &mut [Number]) -> bool {
        for s in x_scaling.iter_mut() {
            *s = 0.0001;
        }
        true
    }

    fn objective_scaling(&self) -> f64 {
        1000.0
    }
}

impl ConstrainedProblem for NLP {
    fn num_constraints(&self) -> usize {
        2
    }
    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        8
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        g_l.swap_with_slice(vec![25.0, 40.0].as_mut_slice());
        g_u.swap_with_slice(vec![2.0e19, 40.0].as_mut_slice());
        true
    }
    fn constraint(&self, x: &[Number], g: &mut [Number]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3] + self.g_offset[0];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] + self.g_offset[1];
        true
    }
    fn constraint_jacobian_indices(&self, irow: &mut [Index], jcol: &mut [Index]) -> bool {
        irow[0] = 0;
        jcol[0] = 0;
        irow[1] = 0;
        jcol[1] = 1;
        irow[2] = 0;
        jcol[2] = 2;
        irow[3] = 0;
        jcol[3] = 3;
        irow[4] = 1;
        jcol[4] = 0;
        irow[5] = 1;
        jcol[5] = 1;
        irow[6] = 1;
        jcol[6] = 2;
        irow[7] = 1;
        jcol[7] = 3;
        true
    }
    fn constraint_jacobian_values(&self, x: &[Number], vals: &mut [Number]) -> bool {
        vals[0] = x[1] * x[2] * x[3]; /* 0,0 */
        vals[1] = x[0] * x[2] * x[3]; /* 0,1 */
        vals[2] = x[0] * x[1] * x[3]; /* 0,2 */
        vals[3] = x[0] * x[1] * x[2]; /* 0,3 */

        vals[4] = 2.0 * x[0]; /* 1,0 */
        vals[5] = 2.0 * x[1]; /* 1,1 */
        vals[6] = 2.0 * x[2]; /* 1,2 */
        vals[7] = 2.0 * x[3]; /* 1,3 */
        true
    }

    // Hessian Implementation
    fn num_hessian_non_zeros(&self) -> usize {
        10
    }
    fn hessian_indices(&self, irow: &mut [Index], jcol: &mut [Index]) -> bool {
        let mut idx = 0;
        for row in 0..4 {
            for col in 0..row + 1 {
                irow[idx] = row;
                jcol[idx] = col;
                idx += 1;
            }
        }
        true
    }
    fn hessian_values(
        &self,
        x: &[Number],
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        vals[0] = obj_factor * 2.0 * x[3]; /* 0,0 */

        vals[1] = obj_factor * x[3]; /* 1,0 */
        vals[2] = 0.0; /* 1,1 */

        vals[3] = obj_factor * x[3]; /* 2,0 */
        vals[4] = 0.0; /* 2,1 */
        vals[5] = 0.0; /* 2,2 */

        vals[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]); /* 3,0 */
        vals[7] = obj_factor * x[0]; /* 3,1 */
        vals[8] = obj_factor * x[0]; /* 3,2 */
        vals[9] = 0.0; /* 3,3 */
        /* add the portion for the first constraint */
        vals[1] += lambda[0] * (x[2] * x[3]); /* 1,0 */

        vals[3] += lambda[0] * (x[1] * x[3]); /* 2,0 */
        vals[4] += lambda[0] * (x[0] * x[3]); /* 2,1 */

        vals[6] += lambda[0] * (x[1] * x[2]); /* 3,0 */
        vals[7] += lambda[0] * (x[0] * x[2]); /* 3,1 */
        vals[8] += lambda[0] * (x[0] * x[1]); /* 3,2 */

        /* add the portion for the second constraint */
        vals[0] += lambda[1] * 2.0; /* 0,0 */

        vals[2] += lambda[1] * 2.0; /* 1,1 */

        vals[5] += lambda[1] * 2.0; /* 2,2 */

        vals[9] += lambda[1] * 2.0; /* 3,3 */
        true
    }

    // The following callback is activated only when `nlp_scaling_method` is set to
    // `user-scaling`.
    fn constraint_scaling(&self, g_scaling: &mut [Number]) -> bool {
        for s in g_scaling.iter_mut() {
            *s = 0.0001;
        }
        true
    }
}

fn hs071() -> Ipopt<NLP> {
    let nlp = NLP {
        g_offset: [0.0, 0.0],
        iterations: 0,
    };
    let mut ipopt = Ipopt::new(nlp).unwrap();
    ipopt.set_option("tol", 1e-7);
    ipopt.set_option("mu_strategy", "adaptive");
    ipopt.set_option("sb", "yes"); // suppress license message
    ipopt.set_option("print_level", 0); // suppress debug output
    ipopt
}

#[test]
fn hs071_test() {
    let mut ipopt = hs071();
    ipopt.set_intermediate_callback(Some(NLP::intermediate_cb));

    // Solve
    let SolveResult {
        solver_data:
            SolverDataMut {
                solution: Solution {
                    primal_variables: x,
                    constraint_multipliers: mult_g,
                    lower_bound_multipliers: mult_x_l,
                    upper_bound_multipliers: mult_x_u,
                },
                ..
            },
        status,
        objective_value: obj,
        ..
    } = ipopt.solve();

    // Check result
    assert_eq!(status, SolveStatus::UserRequestedStop);
    assert_relative_eq!(x[0], 1.000000e+00, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(x[1], 4.743000e+00, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(x[2], 3.821150e+00, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(x[3], 1.379408e+00, max_relative = 1e-6, epsilon = 1e-14);

    assert_relative_eq!(mult_g[0], -5.522936e-01, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_g[1], 1.614685e-01,  max_relative = 1e-6, epsilon = 1e-14);

    assert_relative_eq!(mult_x_l[0], 1.087872e+00, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_x_l[1], 4.635819e-09, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_x_l[2], 9.087447e-09, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_x_l[3], 8.555955e-09, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_x_u[0], 4.470027e-09, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_x_u[1], 4.075231e-07, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_x_u[2], 1.189791e-08, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_x_u[3], 6.398749e-09, max_relative = 1e-6, epsilon = 1e-14);

    assert_relative_eq!(obj, 1.701402e+01, max_relative = 1e-6, epsilon = 1e-14);
}

#[test]
fn hs071_warm_start_test() {
    let mut ipopt = hs071();
    ipopt.set_intermediate_callback(Some(NLP::intermediate_cb));
    {
        let SolveResult {
            solver_data:
                SolverDataMut {
                    problem,
                    ..
                },
            status,
            ..
        } = ipopt.solve();

        assert_eq!(status, SolveStatus::UserRequestedStop);
        assert_eq!(problem.iterations, 7);

        // The results from this tests are checked in the hs071_test.

        problem.g_offset[0] = 0.2;
    }

    ipopt.set_option("warm_start_init_point", "yes");
    ipopt.set_option("bound_push", 1e-5);
    ipopt.set_option("bound_frac", 1e-5);
    ipopt.set_intermediate_callback(Some(NLP::count_iterations_cb));
    {
        let SolveResult {
            solver_data:
                SolverDataMut {
                    problem,
                    solution: Solution {
                        primal_variables: x,
                        constraint_multipliers: mult_g,
                        lower_bound_multipliers: mult_x_l,
                        upper_bound_multipliers: mult_x_u,
                    }
                },
            status,
            objective_value: obj,
            ..
        } = ipopt.solve();

        assert_eq!(status, SolveStatus::SolveSucceeded);
        assert_eq!(problem.iterations, 9);
        assert_relative_eq!(x[0], 1.000000e+00, max_relative = 1e-6, epsilon = 1e-14);
        assert_relative_eq!(x[1], 4.749269e+00, max_relative = 1e-6, epsilon = 1e-14);
        assert_relative_eq!(x[2], 3.817510e+00, max_relative = 1e-6, epsilon = 1e-14);
        assert_relative_eq!(x[3], 1.367870e+00, max_relative = 1e-6, epsilon = 1e-14);

        assert_relative_eq!(mult_g[0], -5.517016e-01, max_relative = 1e-6, epsilon = 1e-14);
        assert_relative_eq!(mult_g[1], 1.592915e-01,  max_relative = 1e-6, epsilon = 1e-14);

        assert_relative_eq!(mult_x_l[0], 1.090362e+00, max_relative = 1e-6, epsilon = 1e-14);
        assert_relative_eq!(mult_x_l[1], 0.0, max_relative = 1e-6, epsilon = 1e-10);
        assert_relative_eq!(mult_x_l[2], 0.0, max_relative = 1e-6, epsilon = 1e-10);
        assert_relative_eq!(mult_x_l[3], 0.0, max_relative = 1e-6, epsilon = 1e-10);
        assert_relative_eq!(mult_x_u[0], 0.0, max_relative = 1e-6, epsilon = 1e-10);
        assert_relative_eq!(mult_x_u[1], 0.0, max_relative = 1e-6, epsilon = 1e-10);
        assert_relative_eq!(mult_x_u[2], 0.0, max_relative = 1e-6, epsilon = 1e-10);
        assert_relative_eq!(mult_x_u[3], 0.0, max_relative = 1e-6, epsilon = 1e-10);

        assert_relative_eq!(obj, 1.690362e+01, max_relative = 1e-6, epsilon = 1e-14);
        problem.g_offset[0] = 0.1;
    }

    // Solve one more time time
    {
        let SolveResult {
            solver_data: SolverDataMut {
                problem,
                ..
            },
            status,
            objective_value: obj,
            ..
        } = ipopt.solve();

        assert_eq!(status, SolveStatus::SolveSucceeded);
        assert_eq!(problem.iterations, 9);
        assert_relative_eq!(obj, 1.695880e+01, epsilon = 1e-5);
    }
}

#[test]
fn hs071_custom_scaling_test() {
    let mut ipopt = hs071();
    ipopt.solver_data_mut().problem.g_offset[0] = 0.2;

    ipopt.set_option("nlp_scaling_method", "user-scaling");
    ipopt.set_intermediate_callback(Some(NLP::count_iterations_cb));

    let SolveResult {
        solver_data:
            SolverDataMut {
                problem,
                solution: Solution {
                    primal_variables: x,
                    constraint_multipliers: mult_g,
                    lower_bound_multipliers: mult_x_l,
                    upper_bound_multipliers: mult_x_u,
                },
            },
        status,
        objective_value: obj,
        ..
    } = ipopt.solve();

    // There is no explicit way to tell if things have been scaled other than looking at the
    // output, however we can check that the number of iterations has changed, and it still
    // converges given all other parameters are the same. This indicates that at least some of our
    // scaling functions are actually being called.
    assert_eq!(status, SolveStatus::SolveSucceeded);
    assert_eq!(problem.iterations, 22);
    assert_relative_eq!(x[0], 1.000000e+00, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(x[1], 4.749269e+00, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(x[2], 3.817510e+00, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(x[3], 1.367870e+00, max_relative = 1e-6, epsilon = 1e-14);

    assert_relative_eq!(mult_g[0], -5.517016e-01, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_g[1], 1.592915e-01,  max_relative = 1e-6, epsilon = 1e-14);

    assert_relative_eq!(mult_x_l[0], 1.090362e+00, max_relative = 1e-6, epsilon = 1e-14);
    assert_relative_eq!(mult_x_l[1], 0.0, max_relative = 1e-6, epsilon = 1e-10);
    assert_relative_eq!(mult_x_l[2], 0.0, max_relative = 1e-6, epsilon = 1e-10);
    assert_relative_eq!(mult_x_l[3], 0.0, max_relative = 1e-6, epsilon = 1e-10);
    assert_relative_eq!(mult_x_u[0], 0.0, max_relative = 1e-6, epsilon = 1e-10);
    assert_relative_eq!(mult_x_u[1], 0.0, max_relative = 1e-6, epsilon = 1e-10);
    assert_relative_eq!(mult_x_u[2], 0.0, max_relative = 1e-6, epsilon = 1e-10);
    assert_relative_eq!(mult_x_u[3], 0.0, max_relative = 1e-6, epsilon = 1e-10);

    assert_relative_eq!(obj, 1.690362e+01, max_relative = 1e-6, epsilon = 1e-14);
}
