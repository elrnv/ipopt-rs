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
extern crate ipopt;
#[macro_use] extern crate approx;

use ipopt::*;

struct NLP {
    g_offset: [f64; 2],
}
impl NLP {
    fn intermediate_cb(
        &mut self,
        _alg_mod: Index,
        _iter_count: Index,
        _obj_value: Number,
        inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index) -> bool
    {
      inf_pr >= 1e-4
    }
}


impl BasicProblem for NLP {
    fn num_variables(&self) -> usize { 4 }
    fn bounds(&self) -> (Vec<Number>, Vec<Number>) { (vec![1.0; 4], vec![5.0; 4]) }
    fn initial_point(&self) -> Vec<Number> { vec![1.0, 5.0, 5.0, 1.0] }
    fn objective(&mut self, x: &[Number], obj: &mut Number) -> bool {
        *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        true
    }
    fn objective_grad(&mut self, x: &[Number], grad_f: &mut [Number]) -> bool {
        grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
        grad_f[1] = x[0] * x[3];
        grad_f[2] = x[0] * x[3] + 1.0;
        grad_f[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }
}

impl ConstrainedProblem for NLP {
    fn num_constraints(&self) -> usize { 2 }
    fn num_constraint_jac_non_zeros(&self) -> usize { 8 }

    fn constraint_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (vec![25.0, 40.0], vec![2.0e19, 40.0])
    }
    fn constraint(&mut self, x: &[Number], g: &mut [Number]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3] + self.g_offset[0];
        g[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] + self.g_offset[1];
        true
    }
    fn constraint_jac_indices(&mut self, irow: &mut [Index], jcol: &mut [Index]) -> bool {
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
    fn constraint_jac_values(&mut self, x: &[Number], vals: &mut [Number]) -> bool {
        vals[0] = x[1]*x[2]*x[3]; /* 0,0 */
        vals[1] = x[0]*x[2]*x[3]; /* 0,1 */
        vals[2] = x[0]*x[1]*x[3]; /* 0,2 */
        vals[3] = x[0]*x[1]*x[2]; /* 0,3 */

        vals[4] = 2.0*x[0];         /* 1,0 */
        vals[5] = 2.0*x[1];         /* 1,1 */
        vals[6] = 2.0*x[2];         /* 1,2 */
        vals[7] = 2.0*x[3];         /* 1,3 */
        true
    }

    // Hessian Implementation
    fn num_hessian_non_zeros(&self) -> usize { 10 }
    fn hessian_indices(&mut self, irow: &mut [Index], jcol: &mut [Index]) -> bool {
        let mut idx = 0;
        for row in 0..4 {
            for col in 0..row+1 {
                irow[idx] = row;
                jcol[idx] = col;
                idx += 1;
            }
        }
        true
    }
    fn hessian_values(&mut self,
                      x: &[Number],
                      obj_factor: Number,
                      lambda: &[Number],
                      vals: &mut [Number]) -> bool {
        vals[0] = obj_factor*2.0*x[3];                  /* 0,0 */

        vals[1] = obj_factor*x[3];                      /* 1,0 */
        vals[2] = 0.0;                                  /* 1,1 */

        vals[3] = obj_factor*x[3];                      /* 2,0 */
        vals[4] = 0.0;                                  /* 2,1 */
        vals[5] = 0.0;                                  /* 2,2 */

        vals[6] = obj_factor*(2.0*x[0] + x[1] + x[2]);  /* 3,0 */
        vals[7] = obj_factor*x[0];                      /* 3,1 */
        vals[8] = obj_factor*x[0];                      /* 3,2 */
        vals[9] = 0.0;                                  /* 3,3 */
        /* add the portion for the first constraint */
        vals[1] += lambda[0] * (x[2] * x[3]);           /* 1,0 */

        vals[3] += lambda[0] * (x[1] * x[3]);           /* 2,0 */
        vals[4] += lambda[0] * (x[0] * x[3]);           /* 2,1 */

        vals[6] += lambda[0] * (x[1] * x[2]);           /* 3,0 */
        vals[7] += lambda[0] * (x[0] * x[2]);           /* 3,1 */
        vals[8] += lambda[0] * (x[0] * x[1]);           /* 3,2 */

        /* add the portion for the second constraint */
        vals[0] += lambda[1] * 2.0;                     /* 0,0 */

        vals[2] += lambda[1] * 2.0;                     /* 1,1 */

        vals[5] += lambda[1] * 2.0;                     /* 2,2 */

        vals[9] += lambda[1] * 2.0;                     /* 3,3 */
        true
    }
}

#[test]
fn hs071_test() {
    let nlp = NLP { g_offset: [0.0, 0.0] };
    let mut ipopt = Ipopt::new(nlp);
    ipopt.set_option("tol", 1e-7);
    ipopt.set_option("mu_strategy", "adaptive");
    ipopt.set_option("sb", "yes"); // suppress license message
    ipopt.set_option("print_level", 0); // suppress debug output
    ipopt.set_intermediate_callback(Some(NLP::intermediate_cb));
    {
        let SolveResult {
            solver_data: SolverData {
                problem,
                primal_variables: x,
                constraint_multipliers: mult_g,
                lower_bound_multipliers: mult_x_l,
                upper_bound_multipliers: mult_x_u,
            },
            status,
            objective_value: obj,
            ..
        } = ipopt.solve();

        assert_eq!(status, SolveStatus::UserRequestedStop);
        assert_relative_eq!(x[0], 1.000000e+00, epsilon = 1e-5);
        assert_relative_eq!(x[1], 4.743000e+00, epsilon = 1e-5);
        assert_relative_eq!(x[2], 3.821150e+00, epsilon = 1e-5);
        assert_relative_eq!(x[3], 1.379408e+00, epsilon = 1e-5);

        assert_relative_eq!(mult_g[0], -5.522936e-01, epsilon = 1e-5);
        assert_relative_eq!(mult_g[1], 1.614685e-01, epsilon = 1e-5);

        assert_relative_eq!(mult_x_l[0], 1.087872e+00, epsilon = 1e-5);
        assert_relative_eq!(mult_x_l[1], 4.635819e-09, epsilon = 1e-5);
        assert_relative_eq!(mult_x_l[2], 9.087447e-09, epsilon = 1e-5);
        assert_relative_eq!(mult_x_l[3], 8.555955e-09, epsilon = 1e-5);
        assert_relative_eq!(mult_x_u[0], 4.470027e-09, epsilon = 1e-5);
        assert_relative_eq!(mult_x_u[1], 4.075231e-07, epsilon = 1e-5);
        assert_relative_eq!(mult_x_u[2], 1.189791e-08, epsilon = 1e-5);
        assert_relative_eq!(mult_x_u[3], 6.398749e-09, epsilon = 1e-5);

        assert_relative_eq!(obj, 1.701402e+01, epsilon = 1e-5);

        problem.g_offset[0] = 0.2;
    }

    ipopt.set_option("warm_start_init_point", "yes");
    ipopt.set_option("bound_push", 1e-5);
    ipopt.set_option("bound_frac", 1e-5);
    ipopt.set_intermediate_callback(None);
    {
        let SolveResult {
            solver_data: SolverData {
                primal_variables: x,
                constraint_multipliers: mult_g,
                lower_bound_multipliers: mult_x_l,
                upper_bound_multipliers: mult_x_u,
                ..
            },
            status,
            objective_value: obj,
            ..
        } = ipopt.solve();

        assert_eq!(status, SolveStatus::SolveSucceeded);
        assert_relative_eq!(x[0], 1.000000e+00, epsilon = 1e-5);
        assert_relative_eq!(x[1], 4.749269e+00, epsilon = 1e-5);
        assert_relative_eq!(x[2], 3.817510e+00, epsilon = 1e-5);
        assert_relative_eq!(x[3], 1.367870e+00, epsilon = 1e-5);

        assert_relative_eq!(mult_g[0], -5.517016e-01, epsilon = 1e-5);
        assert_relative_eq!(mult_g[1], 1.592915e-01, epsilon = 1e-5);

        assert_relative_eq!(mult_x_l[0], 1.090362e+00, epsilon = 1e-5);
        assert_relative_eq!(mult_x_l[1], 2.664877e-12, epsilon = 1e-5);
        assert_relative_eq!(mult_x_l[2], 3.556758e-12, epsilon = 1e-5);
        assert_relative_eq!(mult_x_l[3], 2.693832e-11, epsilon = 1e-5);
        assert_relative_eq!(mult_x_u[0], 2.498100e-12, epsilon = 1e-5);
        assert_relative_eq!(mult_x_u[1], 4.074104e-11, epsilon = 1e-5);
        assert_relative_eq!(mult_x_u[2], 8.423997e-12, epsilon = 1e-5);
        assert_relative_eq!(mult_x_u[3], 2.755724e-12, epsilon = 1e-5);

        assert_relative_eq!(obj, 1.690362e+01, epsilon = 1e-5);
    }
}
