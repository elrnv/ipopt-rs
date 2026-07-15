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

/**
 * This test solves an unconstrained quadratic through the `ConstrainedProblem` interface, with the
 * constraint count reported as zero. This is a valid configuration, and one that arises when the
 * number of constraints is only known at runtime and happens to come out to zero.
 */
use approx::assert_relative_eq;

use ipopt::*;

struct NLP;

impl BasicProblem for NLP {
    fn num_variables(&self) -> usize {
        1
    }
    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        x_l[0] = -1e20;
        x_u[0] = 1e20;
        true
    }
    fn initial_point(&self, x: &mut [Number]) -> bool {
        x[0] = 5.0;
        true
    }
    fn objective(&self, x: &[Number], _: bool, obj: &mut Number) -> bool {
        *obj = (x[0] - 1.0) * (x[0] - 1.0);
        true
    }
    fn objective_grad(&self, x: &[Number], _: bool, grad_f: &mut [Number]) -> bool {
        grad_f[0] = 2.0 * (x[0] - 1.0);
        true
    }
}

impl ConstrainedProblem for NLP {
    fn num_constraints(&self) -> usize {
        0
    }
    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        0
    }
    fn constraint(&self, _: &[Number], _: bool, g: &mut [Number]) -> bool {
        assert!(g.is_empty());
        true
    }
    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        assert!(g_l.is_empty());
        assert!(g_u.is_empty());
        true
    }
    fn constraint_jacobian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        assert!(rows.is_empty());
        assert!(cols.is_empty());
        true
    }
    fn constraint_jacobian_values(&self, _: &[Number], _: bool, vals: &mut [Number]) -> bool {
        assert!(vals.is_empty());
        true
    }

    // Hessian Implementation
    fn num_hessian_non_zeros(&self) -> usize {
        1
    }
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        rows[0] = 0;
        cols[0] = 0;
        true
    }
    fn hessian_values(
        &self,
        _: &[Number],
        _: bool,
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        assert!(lambda.is_empty());
        vals[0] = 2.0 * obj_factor;
        true
    }
}

#[test]
fn zero_constraints_solve_test() {
    let mut ipopt = Ipopt::new(NLP).unwrap();
    ipopt.set_option("tol", 1e-9);
    ipopt.set_option("mu_strategy", "adaptive");
    ipopt.set_option("sb", "yes");
    ipopt.set_option("print_level", 0);

    let SolveResult {
        solver_data:
            SolverDataMut {
                solution:
                    Solution {
                        primal_variables: x,
                        constraint_multipliers,
                        ..
                    },
                ..
            },
        constraint_values,
        objective_value: obj,
        status,
    } = ipopt.solve();

    assert_eq!(status, SolveStatus::SolveSucceeded);
    assert_relative_eq!(x[0], 1.0, epsilon = 1e-8);
    assert_relative_eq!(obj, 0.0, epsilon = 1e-8);
    assert!(constraint_multipliers.is_empty());
    assert!(constraint_values.is_empty());
}
