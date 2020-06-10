//   Copyright 2020 Egor Larionov
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

//! In this example we demonstrate how a quadratic function can be minimized subject to linear and
//! non-linear constraints.
//!
//! Additionally, this example shows how discontinuities in the constraint Jacobian can cause
//! instabilities and ultimatily a failed optimization.

use ipopt::*;

/// Non-linear problem to be solved.
struct NLP<C, J, H> {
    /// Constraint function.
    ///
    /// This function is a map from R^2 to R.
    pub constraint_f: C,
    /// Constraint Jacobian.
    pub constraint_jac: J,
    /// Constraint Hessian.
    pub constraint_hess: H,
    /// Keep track of the number of iterations for each solve.
    pub iterations: usize,
}

impl<C, J, H> NLP<C, J, H> {
    fn count_iterations_cb(&mut self, data: IntermediateCallbackData) -> bool {
        self.iterations = data.iter_count as usize;
        true
    }
}

impl<C, J, H> BasicProblem for NLP<C, J, H> {
    fn num_variables(&self) -> usize {
        2
    }
    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        x_l[0] = -1e20;
        x_l[1] = -1e20;
        x_u[0] = 1e20;
        x_u[1] = 1e20;
        true
    }
    fn initial_point(&self, x: &mut [Number]) -> bool {
        x[0] = 0.5;
        x[1] = 0.8;
        true
    }
    fn objective(&self, x: &[Number], obj: &mut Number) -> bool {
        *obj = quadratic(x[0], x[1]);
        true
    }
    fn objective_grad(&self, x: &[Number], grad_f: &mut [Number]) -> bool {
        let grad = quadratic_grad(x[0], x[1]);
        grad_f[0] = grad[0];
        grad_f[1] = grad[1];
        true
    }
}

impl<C, J, H> ConstrainedProblem for NLP<C, J, H>
where C: Fn(f64, f64) -> f64,
      J: Fn(f64, f64) -> [f64; 2],
      H: Fn(f64, f64) -> [f64; 3], // Lower triangular [H00, H11, H10]
{
    fn num_constraints(&self) -> usize {
        1
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        2
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        g_l[0] = 0.0;
        g_u[0] = 1e20;
        true
    }
    fn constraint(&self, x: &[Number], g: &mut [Number]) -> bool {
        g[0] = (self.constraint_f)(x[0], x[1]);
        true
    }
    fn constraint_jacobian_indices(&self, irow: &mut [Index], jcol: &mut [Index]) -> bool {
        irow[0] = 0;
        jcol[0] = 0;
        irow[1] = 0;
        jcol[1] = 1;
        true
    }
    fn constraint_jacobian_values(&self, x: &[Number], vals: &mut [Number]) -> bool {
        let jac = (self.constraint_jac)(x[0], x[1]);
        vals[0] = jac[0];
        vals[1] = jac[1];
        true
    }
    fn num_hessian_non_zeros(&self) -> usize {
        // In reality the Hessian is rather sparse, but we will include all entries for simplicity.
        // Sparsity should be exploited here for performance in real world problems.
        3 + 3 // 3 for objective Hessian and 3 for constraint Hessian.
    }
    fn hessian_indices(&self, irow: &mut [Index], jcol: &mut [Index]) -> bool {
        // Objective Hessian entries:
        irow[0] = 0;
        jcol[0] = 0;

        irow[1] = 1;
        jcol[1] = 1;

        irow[2] = 1;
        jcol[2] = 0;

        // Constraint Hessian entries:
        irow[3] = 0;
        jcol[3] = 0;

        irow[4] = 1;
        jcol[4] = 1;

        irow[5] = 1;
        jcol[5] = 0;
        true
    }
    fn hessian_values(&self, x: &[Number], obj_factor: Number, lambda: &[Number], vals: &mut [Number]) -> bool {
        let obj_hess = quadratic_hessian(x[0], x[1]);
        let constraint_hess = (self.constraint_hess)(x[0], x[1]);
        vals[0] = obj_hess[0] * obj_factor;
        vals[1] = obj_hess[1] * obj_factor;
        vals[2] = obj_hess[2] * obj_factor;

        vals[3] = constraint_hess[0] * lambda[0];
        vals[4] = constraint_hess[1] * lambda[0];
        vals[5] = constraint_hess[2] * lambda[0];
        true
    }
}

/// A signed distance field to the absolute value function.
///
/// This function has a cusp at `x = 0`, where it is not differentiable for any `y`.
fn abs_sdf(x: f64, y: f64) -> f64 {
    (y-x).min(x+y)/2.0_f64.sqrt()
}

/// Jacobian of the `abs_sdf` function.
fn abs_sdf_jacobian(x: f64, _y: f64) -> [f64; 2] {
    // Disambiguate the jacobian at x=0 to coincide with the jacobian in x > 0.
    let sqrt2 = 2.0_f64.sqrt();
    if x > 0.0 {
        [-1.0/sqrt2, 1.0/sqrt2]
    } else {
        [1.0/sqrt2, 1.0/sqrt2]
    }
}

// Note that the hessian of `abs_sdf` is zero.

/// A smoothed version of the absolute value function field `abs_sdf`.
fn smoothed_abs_sdf(x: f64, y: f64, eps: f64) -> f64 {
    if -x.abs() + 2.0*eps > y {
        // Near the cusp we use a circle arc as an approximation.
        let y_minus_2_eps = y - 2.0 * eps;
        eps*2.0_f64.sqrt() - (x*x + y_minus_2_eps * y_minus_2_eps).sqrt()
    } else {
        abs_sdf(x, y)
    }
}

/// Jacobian of the `smoothed_abs_sdf` function.
fn smoothed_abs_sdf_jacobian(x: f64, y: f64, eps: f64) -> [f64; 2] {
    if -x.abs() + 2.0*eps > y {
        let y_minus_2_eps = y - 2.0 * eps;
        let factor = -1.0/(x*x + y_minus_2_eps * y_minus_2_eps).sqrt();
        [factor * x, factor * y_minus_2_eps]
    } else {
        abs_sdf_jacobian(x, y)
    }
}

/// The lower triangular part of the Hessian of the `smoothed_abs_sdf` function.
///
/// The returned array contains the lower triangular entries [H00, H11, H10].
fn smoothed_abs_sdf_hessian(x: f64, y: f64, eps: f64) -> [f64; 3] {
    if -x.abs() + 2.0*eps > y {
        let y_minus_2_eps = y - 2.0 * eps;
        let f = x*x + y_minus_2_eps * y_minus_2_eps;
        let factor1 = -1.0/f.sqrt();
        let factor2 = 1.0/(f*f.sqrt());
        [factor1 + factor2 * x * x, factor1 + factor2 * y_minus_2_eps * y_minus_2_eps, factor2 * y_minus_2_eps * x]
    } else {
        [0.0; 3]
    }
}

/// The quadratic function to be minimized.
///
/// This function attains a minimum at (0, -1), and increases parabolically away from that point.
fn quadratic(x: f64, y: f64) -> f64 {
    0.25 * (x * x + (y + 1.0) * (y + 1.0))
}

/// The gradient of the `quadratic`.
fn quadratic_grad(x: f64, y: f64) -> [f64; 2] {
    [0.5 * x, 0.5 * (y + 1.0)]
}

/// The lower triangular part of the Hessian of the `quadratic`.
///
/// The returned array contains the lower triangular entries [H00, H11, H10].
fn quadratic_hessian(_x: f64, _y: f64) -> [f64; 3] {
    [ 0.5, 0.5, 0.0 ]
}

fn main() {
    // First, let us try to minimize the `quadratic` subject to the non-smooth constraint
    // `abs_sdf(x,y) => 0`.
    let non_smooth_constraint_nlp = NLP {
        constraint_f: abs_sdf,
        constraint_jac: abs_sdf_jacobian,
        constraint_hess: |_, _| [0.0; 3],
        iterations: 0,
    };

    let mut ipopt = Ipopt::new(non_smooth_constraint_nlp).unwrap();
    ipopt.set_intermediate_callback(Some(NLP::count_iterations_cb));
    ipopt.set_option("tol", 1e-7);
    ipopt.set_option("mu_strategy", "adaptive");
    ipopt.set_option("print_level", 5);
    let max_iter = 1000;
    ipopt.set_option("max_iter", max_iter as i32);
    let SolveResult {
        solver_data: SolverDataMut {
            problem,
            solution: Solution {
                primal_variables: x,
                ..
            },
        },
        status,
        ..
    } = ipopt.solve();

    assert_eq!(problem.iterations, max_iter);
    assert_eq!(status, SolveStatus::MaximumIterationsExceeded);
    eprintln!("Solve did not converge after {} steps.", max_iter);
    eprintln!("The final result was x: {:?}", x);
    
    // We could replace the single absolute value constraint with two linear constraints to fix
    // convergence.  However this is not always desirable or possible with more complex functions.
    // So in this example, to improve convergence, we will smooth the cusp slightly, preserving the
    // total number of constraints.
    // Try smaller values of epsilon to see at which point it starts to affect convergence.
    let epsilon = 0.1;
    let smooth_constraint_nlp = NLP {
        constraint_f: |x, y| smoothed_abs_sdf(x, y, epsilon),
        constraint_jac: |x, y| smoothed_abs_sdf_jacobian(x, y, epsilon),
        constraint_hess: |x, y| smoothed_abs_sdf_hessian(x, y, epsilon),
        iterations: 0,
    };

    let mut ipopt = Ipopt::new(smooth_constraint_nlp).unwrap();
    ipopt.set_intermediate_callback(Some(NLP::count_iterations_cb));
    ipopt.set_option("tol", 1e-7);
    ipopt.set_option("mu_strategy", "adaptive");
    ipopt.set_option("print_level", 5);

    let SolveResult {
        solver_data: SolverDataMut {
            problem,
            solution: Solution {
                primal_variables: x,
                ..
            },
        },
        ..
    } = ipopt.solve();

    assert!(problem.iterations < 10);
    eprintln!("Solve converged after less than 10 iterations at x: {:?}", x);
}
