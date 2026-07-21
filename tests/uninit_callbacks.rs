//   Copyright 2026 Egor Larionov
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

//! Solves the HS071 problem, but implements the hot evaluation callbacks through the zero-overhead
//! `*_uninit` variants instead of the safe `&mut [Number]` methods.
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use approx::assert_relative_eq;

use ipopt::*;

fn objective(x: &[Number]) -> Number {
    x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
}

fn grad(x: &[Number]) -> [Number; 4] {
    [
        x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
        x[0] * x[3],
        x[0] * x[3] + 1.0,
        x[0] * (x[0] + x[1] + x[2]),
    ]
}

fn constraint_vals(x: &[Number]) -> [Number; 2] {
    [
        x[0] * x[1] * x[2] * x[3],
        x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3],
    ]
}

fn jac(x: &[Number]) -> [Number; 8] {
    [
        x[1] * x[2] * x[3],
        x[0] * x[2] * x[3],
        x[0] * x[1] * x[3],
        x[0] * x[1] * x[2],
        2.0 * x[0],
        2.0 * x[1],
        2.0 * x[2],
        2.0 * x[3],
    ]
}

fn hess(x: &[Number], obj_factor: Number, lambda: &[Number]) -> [Number; 10] {
    let mut h = [0.0; 10];
    h[0] = obj_factor * 2.0 * x[3];
    h[1] = obj_factor * x[3];
    h[3] = obj_factor * x[3];
    h[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]);
    h[7] = obj_factor * x[0];
    h[8] = obj_factor * x[0];

    h[1] += lambda[0] * (x[2] * x[3]);
    h[3] += lambda[0] * (x[1] * x[3]);
    h[4] += lambda[0] * (x[0] * x[3]);
    h[6] += lambda[0] * (x[1] * x[2]);
    h[7] += lambda[0] * (x[0] * x[2]);
    h[8] += lambda[0] * (x[0] * x[1]);

    h[0] += lambda[1] * 2.0;
    h[2] += lambda[1] * 2.0;
    h[5] += lambda[1] * 2.0;
    h[9] += lambda[1] * 2.0;
    h
}

#[derive(Default)]
struct Hs071Uninit {
    /// Counts how many times a `*_uninit` override actually ran.
    uninit_calls: AtomicUsize,
}

impl BasicProblem for Hs071Uninit {
    fn num_variables(&self) -> usize {
        4
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        x_l.copy_from_slice(&[1.0; 4]);
        x_u.copy_from_slice(&[5.0; 4]);
        true
    }

    fn initial_point(&self, x: &mut [Number]) -> bool {
        x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
        true
    }

    fn objective(&self, x: &[Number], _: bool, obj: &mut Number) -> bool {
        *obj = objective(x);
        true
    }

    fn objective_grad(&self, x: &[Number], _: bool, grad_f: &mut [Number]) -> bool {
        grad_f.copy_from_slice(&grad(x));
        true
    }

    fn objective_grad_uninit<'g>(
        &self,
        x: &[Number],
        _: bool,
        grad_f: &'g mut [MaybeUninit<Number>],
    ) -> Option<&'g mut [Number]> {
        self.uninit_calls.fetch_add(1, Ordering::Relaxed);
        let g = grad(x);
        grad_f[0].write(g[0]);
        grad_f[1].write(g[1]);
        grad_f[2].write(g[2]);
        grad_f[3].write(g[3]);
        // SAFETY: every element of `grad_f` was written above.
        Some(unsafe { slice_assume_init_mut(grad_f) })
    }
}

impl ConstrainedProblem for Hs071Uninit {
    fn num_constraints(&self) -> usize {
        2
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        8
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        g_l.copy_from_slice(&[25.0, 40.0]);
        g_u.copy_from_slice(&[2.0e19, 40.0]);
        true
    }

    fn constraint(&self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&constraint_vals(x));
        true
    }

    fn constraint_uninit<'g>(
        &self,
        x: &[Number],
        _: bool,
        g: &'g mut [MaybeUninit<Number>],
    ) -> Option<&'g mut [Number]> {
        self.uninit_calls.fetch_add(1, Ordering::Relaxed);
        let c = constraint_vals(x);
        Some(g.init_from_fn(|i| c[i]))
    }

    fn constraint_jacobian_indices(&self, irow: &mut [Index], jcol: &mut [Index]) -> bool {
        let rows = [0, 0, 0, 0, 1, 1, 1, 1];
        let cols = [0, 1, 2, 3, 0, 1, 2, 3];
        irow.copy_from_slice(&rows);
        jcol.copy_from_slice(&cols);
        true
    }

    fn constraint_jacobian_values(&self, x: &[Number], _: bool, vals: &mut [Number]) -> bool {
        vals.copy_from_slice(&jac(x));
        true
    }

    fn constraint_jacobian_values_uninit<'v>(
        &self,
        x: &[Number],
        _: bool,
        vals: &'v mut [MaybeUninit<Number>],
    ) -> Option<&'v mut [Number]> {
        self.uninit_calls.fetch_add(1, Ordering::Relaxed);
        Some(vals.write_copy_of_slice(&jac(x)))
    }

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
        _: bool,
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        vals.copy_from_slice(&hess(x, obj_factor, lambda));
        true
    }

    fn hessian_values_uninit<'v>(
        &self,
        x: &[Number],
        _: bool,
        obj_factor: Number,
        lambda: &[Number],
        vals: &'v mut [MaybeUninit<Number>],
    ) -> Option<&'v mut [Number]> {
        self.uninit_calls.fetch_add(1, Ordering::Relaxed);
        Some(vals.write_copy_of_slice(&hess(x, obj_factor, lambda)))
    }
}

#[test]
fn uninit_callbacks_solve() {
    let mut ipopt = Ipopt::new(Hs071Uninit::default()).unwrap();
    ipopt.set_option("tol", 1e-7);
    ipopt.set_option("mu_strategy", "adaptive");
    ipopt.set_option("sb", "yes");
    ipopt.set_option("print_level", 0);

    let SolveResult {
        solver_data:
            SolverDataMut {
                problem,
                solution:
                    Solution {
                        primal_variables: x,
                        ..
                    },
            },
        status,
        objective_value: obj,
        ..
    } = ipopt.solve();

    assert_eq!(status, SolveStatus::SolveSucceeded);
    assert!(problem.uninit_calls.load(Ordering::Relaxed) > 0);

    assert_relative_eq!(x[0], 1.000000e+00, max_relative = 1e-6);
    assert_relative_eq!(x[1], 4.743000e+00, max_relative = 1e-6);
    assert_relative_eq!(x[2], 3.821150e+00, max_relative = 1e-6);
    assert_relative_eq!(x[3], 1.379408e+00, max_relative = 1e-6);
    assert_relative_eq!(obj, 1.701402e+01, max_relative = 1e-6);
}
