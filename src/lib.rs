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

#![warn(missing_docs)]

/*!
 * # ipopt-rs
 *
 * This crate provides a safe Rust interface to the [Ipopt](https://projects.coin-or.org/Ipopt)
 * non-linear optimization library. From the Ipopt webpage:
 *
 * > Ipopt (**I**nterior **P**oint **OPT**imizer, pronounced eye-pea-Opt) is a software package
 * > for large-scale nonlinear optimization. It is designed to find (local) solutions of
 * > mathematical optimization problems of the from
 * >
 * >```verbatim
 * >    min     f(x)
 * >    x in R^n
 * >
 * >    s.t.       g_L <= g(x) <= g_U
 * >               x_L <=  x   <= x_U
 * >```
 * >
 * > where `f(x): R^n --> R` is the objective function, and `g(x): R^n --> R^m` are the
 * > constraint functions. The vectors `g_L` and `g_U` denote the lower and upper bounds
 * > on the constraints, and the vectors `x_L` and `x_U` are the bounds on the variables
 * > `x`. The functions `f(x)` and `g(x)` can be nonlinear and nonconvex, but should be
 * > twice continuously differentiable. Note that equality constraints can be
 * > formulated in the above formulation by setting the corresponding components of
 * > `g_L` and `g_U` to the same value.
 *
 * This crate aims to
 *   - reduce the boilerplate, especially for setting up simple unconstrained problems, and
 *   - prevent common mistakes when defining optimization problems as early as possible.
 *
 *
 * # Examples
 *
 * Solve a simple unconstrained problem using L-BFGS: minimize `(x - 1)^2 + (y -1)^2`
 *
 *
 * ```
 * use approx::*;
 * use ipopt::*;
 *
 * struct NLP {
 * }
 *
 * impl BasicProblem for NLP {
 *     // There are two independent variables: x and y.
 *     fn num_variables(&self) -> usize {
 *         2
 *     }    
 *     // The variables are unbounded. Any lower bound lower than -10^19 and upper bound higher
 *     // than 10^19 is treated effectively as infinity. These absolute infinity limits can be
 *     // changed via the `nlp_lower_bound_inf` and `nlp_upper_bound_inf` Ipopt options.
 *     fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
 *         x_l.swap_with_slice(vec![-1e20; 2].as_mut_slice());
 *         x_u.swap_with_slice(vec![1e20; 2].as_mut_slice());
 *         true
 *     }
 *
 *     // Set the initial conditions for the solver.
 *     fn initial_point(&self, x: &mut [Number]) -> bool {
 *         x.swap_with_slice(vec![0.0, 0.0].as_mut_slice());
 *         true
 *     }
 *
 *     // The objective to be minimized.
 *     fn objective(&self, x: &[Number], obj: &mut Number) -> bool {
 *         *obj = (x[0] - 1.0)*(x[0] - 1.0) + (x[1] - 1.0)*(x[1] - 1.0);
 *         true
 *     }
 *
 *     // Objective gradient is used to find a new search direction to find the critical point.
 *     fn objective_grad(&self, x: &[Number], grad_f: &mut [Number]) -> bool {
 *         grad_f[0] = 2.0*(x[0] - 1.0);
 *         grad_f[1] = 2.0*(x[1] - 1.0);
 *         true
 *     }
 * }
 *
 * fn main() {
 *     let nlp = NLP { };
 *     let mut ipopt = Ipopt::new_unconstrained(nlp).unwrap();
 *
 *     // Set Ipopt specific options here a list of all options is available at
 *     // https://www.coin-or.org/Ipopt/documentation/node40.html
 *     ipopt.set_option("tol", 1e-9); // set error tolerance
 *     ipopt.set_option("print_level", 5); // set the print level (5 is the default)
 *
 *     let solve_result = ipopt.solve();
 *
 *     assert_eq!(solve_result.status, SolveStatus::SolveSucceeded);
 *     assert_relative_eq!(solve_result.objective_value, 0.0, epsilon = 1e-10);
 *     let solution = solve_result.solver_data.solution;
 *     assert_relative_eq!(solution.primal_variables[0], 1.0, epsilon = 1e-10);
 *     assert_relative_eq!(solution.primal_variables[1], 1.0, epsilon = 1e-10);
 * }
 * ```
 *
 * See the tests for more examples including constrained optimization.
 *
 */

use ipopt_sys as ffi;

use crate::ffi::{CNLP_Bool as Bool, CNLP_Int as Int};
pub use crate::ffi::{
    // Index type used to access internal buffers.
    CNLP_Index as Index, // i32
    // Uniform floating point number type.
    CNLP_Number as Number, // f64
};

use std::ffi::CString;
use std::fmt::{Debug, Display, Formatter};
use std::slice;

/// The non-linear problem to be solved by Ipopt. This trait specifies all the
/// information needed to construct the unconstrained optimization problem (although the
/// variables are allowed to be bounded).
/// In the callbacks within, `x` is the independent variable and must be the same size
/// as returned by `num_variables`.
/// Each of the callbacks required during interior point iterations are allowed to fail.
/// In case of failure to produce values, simply return `false` where applicable. If the caller
/// returns `true` but the output data was not set, then Ipopt may produce undefined behaviour.
/// This feature could be used to tell Ipopt to try smaller perturbations for `x` for
/// instance.
pub trait BasicProblem {
    /// Specify the indexing style used for arrays in this problem.
    /// (Default is zero-based)
    fn indexing_style(&self) -> IndexingStyle {
        IndexingStyle::CStyle
    }
    /// Total number of variables of the non-linear problem.
    fn num_variables(&self) -> usize;

    /// Specify lower and upper variable bounds given by `x_l` and `x_u` respectively.
    /// Both slices will have the same size as what `num_variables` returns.
    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool;

    /// Construct the initial guess of the primal variables for Ipopt to start with.
    /// The given slice has the same size as `num_variables`.
    ///
    /// This function should return whether the output slice `x` has been populated with initial
    /// values. If this function returns `false`, then a zero initial guess will be used.
    fn initial_point(&self, x: &mut [Number]) -> bool;

    /// Construct the initial guess of the lower and upper bounds multipliers for Ipopt to start with.
    /// The given slices has the same size as `num_variables`.
    /// Note that multipliers for infinity bounds are ignored.
    ///
    /// This function should return whether the output slices `z_l` and `z_u` have been populated
    /// with initial values. If this function returns `false`, then a zero initial guess will be used.
    ///
    /// For convenience, the default implementation initializes bounds multipliers to zero. This is
    /// a good guess for any initial point in the interior of the feasible region.
    fn initial_bounds_multipliers(&self, z_l: &mut [Number], z_u: &mut [Number]) -> bool {
        for (l, u) in z_l.iter_mut().zip(z_u.iter_mut()) {
            *l = 0.0;
            *u = 0.0;
        }
        true
    }

    /// Objective function. This is the function being minimized.
    /// This function is internally called by Ipopt callback `eval_f`.
    fn objective(&self, x: &[Number], obj: &mut Number) -> bool;

    /// Gradient of the objective function.
    /// This function is internally called by Ipopt callback `eval_grad_f`.
    fn objective_grad(&self, x: &[Number], grad_f: &mut [Number]) -> bool;

    /// Provide custom variable scaling.
    /// This method is called if the Ipopt option, `nlp_scaling_method`, is set to `user-scaling`.
    /// Return `true` if scaling is provided and `false` otherwise: if `false` is returned, Ipopt
    /// will not scale the variables.
    ///
    /// The dimension of `x_scaling` is given by `num_variables`.
    ///
    /// For convenience, this function returns `false` by default without modifying the `x_scaling`
    /// slice.
    fn variable_scaling(&self, _x_scaling: &mut [Number]) -> bool {
        false
    }

    /// Provide custom scaling for the objective function.
    /// This method is called if the Ipopt option, `nlp_scaling_method`, is set to `user-scaling`.
    /// For example if this function returns `10`, then then Ipopt solves internally an optimization
    /// problem that has 10 times the value of the original objective function. If this function
    /// returns `-1.0`, then Ipopt will maximize the objective instead of minimizing it.
    ///
    /// For convenience, this function returns `1.0` by default.
    fn objective_scaling(&self) -> f64 {
        1.0
    }
}

/// An extension to the `BasicProblem` trait that enables full Newton iterations in
/// Ipopt. If this trait is NOT implemented by your problem, Ipopt will be set to perform
/// [Quasi-Newton Approximation](https://www.coin-or.org/Ipopt/documentation/node31.html)
/// for second derivatives.
/// This interface asks for the Hessian matrix in sparse triplet form.
pub trait NewtonProblem: BasicProblem {
    /// Number of non-zeros in the Hessian matrix.
    fn num_hessian_non_zeros(&self) -> usize;
    /// Hessian indices. These are the row and column indices of the non-zeros
    /// in the sparse representation of the matrix.
    /// This is a symmetric matrix, fill the lower left triangular half only.
    /// If your problem is constrained (i.e. you are ultimately implementing
    /// `ConstrainedProblem`), ensure that you provide coordinates for non-zeros of the
    /// constraint hessian as well.
    /// This function is internally called by Ipopt callback `eval_h`.
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Objective Hessian values. Each value must correspond to the `row` and `column` as
    /// specified in `hessian_indices`.
    /// This function is internally called by Ipopt callback `eval_h` and each value is
    /// premultiplied by `Ipopt`'s `obj_factor` as necessary.
    fn hessian_values(&self, x: &[Number], vals: &mut [Number]) -> bool;
}

/// Extends the `BasicProblem` trait to enable equality and inequality constraints.
/// Equality constraints are enforced by setting the lower and upper bounds for the
/// constraint to the same value.
/// This type of problem is the target use case for Ipopt.
/// NOTE: Although it's possible to run Quasi-Newton iterations on a constrained problem,
/// it doesn't perform well in general, which is the reason why you must also provide the
/// Hessian callbacks.  However, you may still enable L-BFGS explicitly by setting the
/// "hessian_approximation" Ipopt option to "limited-memory", in which case you should
/// simply return `false` in `hessian_indices` and `hessian_values`.
pub trait ConstrainedProblem: BasicProblem {
    /// Number of equality and inequality constraints.
    fn num_constraints(&self) -> usize;
    /// Number of non-zeros in the constraint Jacobian.
    fn num_constraint_jacobian_non_zeros(&self) -> usize;
    /// Constraint function. This gives the value of each constraint.
    /// The output slice `g` is guaranteed to be the same size as `num_constraints`.
    /// This function is internally called by Ipopt callback `eval_g`.
    fn constraint(&self, x: &[Number], g: &mut [Number]) -> bool;
    /// Specify lower and upper bounds, `g_l` and `g_u` respectively, on the constraint function.
    /// Both slices will have the same size as what `num_constraints` returns.
    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool;
    /// Construct the initial guess of the constraint multipliers for Ipopt to start with.
    /// The given slice has the same size as `num_constraints`.
    ///
    /// This function should return whether the output slice `lambda` has been populated with
    /// initial values. If this function returns `false`, then a zero initial guess will be used.
    ///
    /// For convenience, the default implementation initializes constraint multipliers to zero.
    /// This is a good guess for any initial point in the interior of the feasible region.
    fn initial_constraint_multipliers(&self, lambda: &mut [Number]) -> bool {
        for l in lambda.iter_mut() {
            *l = 0.0;
        }
        true
    }
    /// Constraint Jacobian indices. These are the row and column indices of the
    /// non-zeros in the sparse representation of the matrix.
    /// This function is internally called by Ipopt callback `eval_jac_g`.
    fn constraint_jacobian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Constraint Jacobian values. Each value must correspond to the `row` and
    /// `column` as specified in `constraint_jacobian_indices`.
    /// This function is internally called by Ipopt callback `eval_jac_g`.
    fn constraint_jacobian_values(&self, x: &[Number], vals: &mut [Number]) -> bool;
    /// Number of non-zeros in the Hessian matrix. This includes the constraint hessian.
    fn num_hessian_non_zeros(&self) -> usize;
    /// Hessian indices. These are the row and column indices of the non-zeros
    /// in the sparse representation of the matrix.
    /// This should be a symmetric matrix, fill the lower left triangular half only.
    /// Ensure that you provide coordinates for non-zeros of the
    /// objective and constraint hessians.
    /// This function is internally called by Ipopt callback `eval_h`.
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Hessian values. Each value must correspond to the `row` and `column` as
    /// specified in `hessian_indices`.
    /// Write the objective hessian values multiplied by `obj_factor` and constraint
    /// hessian values multipled by the corresponding values in `lambda` (the Lagrange
    /// multiplier).
    /// This function is internally called by Ipopt callback `eval_h`.
    fn hessian_values(
        &self,
        x: &[Number],
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool;

    /// Provide custom constraint function scaling.
    /// This method is called if the Ipopt option, `nlp_scaling_method`, is set to `user-scaling`.
    /// Return `true` if scaling is provided and `false` otherwise: if `false` is returned, Ipopt
    /// will not scale the constraint function.
    ///
    /// `g_scaling` has the same dimensions as the constraint function: the value returned by
    /// `num_constraints`.
    ///
    /// For convenience, this function returns `false` by default without modifying the `g_scaling`
    /// slice.
    fn constraint_scaling(&self, _g_scaling: &mut [Number]) -> bool {
        false
    }
}

/// Type of option you can specify to Ipopt.
/// This is used internally for conversion.
pub enum IpoptOption<'a> {
    /// Numeric option.
    Num(f64),
    /// String option.
    Str(&'a str),
    /// Integer option.
    Int(i32),
}

/// Convert a floating point value to an `IpoptOption`.
impl<'a> From<f64> for IpoptOption<'a> {
    fn from(opt: f64) -> Self {
        IpoptOption::Num(opt)
    }
}

/// Convert a string to an `IpoptOption`.
impl<'a> From<&'a str> for IpoptOption<'a> {
    fn from(opt: &'a str) -> Self {
        IpoptOption::Str(opt)
    }
}

/// Convert an integer value to an `IpoptOption`.
impl<'a> From<i32> for IpoptOption<'a> {
    fn from(opt: i32) -> Self {
        IpoptOption::Int(opt)
    }
}

/// The solution of the optimization problem including variables, bound multipliers and Lagrange
/// multipliers. This struct stores immutable slices to the solution data.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution<'a> {
    /// This is the solution after the solve.
    pub primal_variables: &'a [Number],
    /// Lower bound multipliers.
    pub lower_bound_multipliers: &'a [Number],
    /// Upper bound multipliers.
    pub upper_bound_multipliers: &'a [Number],
    /// Constraint multipliers, which are available only from contrained problems.
    pub constraint_multipliers: &'a [Number],
}

impl<'a> Solution<'a> {
    /// Construct the solution from raw arrays returned from the Ipopt C interface.
    fn from_raw(
        data: ffi::CNLP_SolverData,
        num_primal_vars: usize,
        num_dual_vars: usize,
    ) -> Solution<'a> {
        Solution {
            primal_variables: unsafe { slice::from_raw_parts(data.x, num_primal_vars) },
            lower_bound_multipliers: unsafe {
                slice::from_raw_parts(data.mult_x_L, num_primal_vars)
            },
            upper_bound_multipliers: unsafe {
                slice::from_raw_parts(data.mult_x_U, num_primal_vars)
            },
            constraint_multipliers: unsafe { slice::from_raw_parts(data.mult_g, num_dual_vars) },
        }
    }
}

/// An interface to mutably access the input problem
/// which Ipopt owns. This method also returns the solver paramters as immutable.
#[derive(Debug, PartialEq)]
pub struct SolverDataMut<'a, P: 'a> {
    /// A mutable reference to the original input problem.
    pub problem: &'a mut P,
    /// Argument solution to the optimization problem.
    pub solution: Solution<'a>,
}

/// An interface to access internal solver data including the input problem immutably.
#[derive(Clone, Debug, PartialEq)]
pub struct SolverData<'a, P: 'a> {
    /// A mutable reference to the original input problem.
    pub problem: &'a P,
    /// Argument solution to the optimization problem.
    pub solution: Solution<'a>,
}

/// Enum that indicates in which mode the algorithm is at some point in time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AlgorithmMode {
    /// Ipopt is in regular mode.
    Regular,
    /// Ipopt is in restoration phase. See Ipopt documentation for details.
    RestorationPhase,
}

/// Pieces of solver data available from Ipopt after each iteration inside the intermediate
/// callback.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IntermediateCallbackData {
    /// Algorithm mode indicates which mode the algorithm is currently in.
    pub alg_mod: AlgorithmMode,
    /// The current iteration count. This includes regular iterations and iterations during the
    /// restoration phase.
    pub iter_count: Index,
    /// The unscaled objective value at the current point. During the restoration phase, this value
    /// remains the unscaled objective value for the original problem.
    pub obj_value: Number,
    /// The unscaled constraint violation at the current point. This quantity is the infinity-norm
    /// (max) of the (unscaled) constraints. During the restoration phase, this value remains
    /// the constraint violation of the original problem at the current point. The option
    /// inf_pr_output can be used to switch to the printing of a different quantity.
    pub inf_pr: Number,
    /// The scaled dual infeasibility at the current point. This quantity measures the infinity-norm
    /// (max) of the internal dual infeasibility, Eq. (4a) in the [implementation
    /// paper](https://www.coin-or.org/Ipopt/documentation/node64.html#WaecBieg06:mp),
    /// including inequality constraints reformulated using slack variables and problem scaling.
    /// During the restoration phase, this is the value of the dual infeasibility for the
    /// restoration phase problem.
    pub inf_du: Number,
    /// The value of the barrier parameter $ \mu$.
    pub mu: Number,
    /// The infinity norm (max) of the primal step (for the original variables $ x$ and the
    /// internal slack variables $ s$). During the restoration phase, this value includes the
    /// values of additional variables, $ p$ and $ n$ (see Eq. (30) in [the implementation
    /// paper](https://www.coin-or.org/Ipopt/documentation/node64.html#WaecBieg06:mp))
    pub d_norm: Number,
    /// The value of the regularization term for the Hessian of the Lagrangian in
    /// the augmented system ($ \delta_w$ in Eq. (26) and Section 3.1 in [the implementation
    /// paper](https://www.coin-or.org/Ipopt/documentation/node64.html#WaecBieg06:mp)). A zero
    /// value indicates that no regularization was done.
    pub regularization_size: Number,
    /// The stepsize for the dual variables ( $ \alpha^z_k$ in Eq. (14c) in [the implementation
    /// paper](https://www.coin-or.org/Ipopt/documentation/node64.html#WaecBieg06:mp)).
    pub alpha_du: Number,
    /// The stepsize for the primal variables ($ \alpha_k$ in Eq. (14a) in [the implementation
    /// paper](https://www.coin-or.org/Ipopt/documentation/node64.html#WaecBieg06:mp)).
    pub alpha_pr: Number,
    /// The number of backtracking line search steps (does not include second-order correction steps).
    pub ls_trials: Index,
}

/// A data structure to store data returned by the solver.
#[derive(Debug, PartialEq)]
pub struct SolveResult<'a, P: 'a> {
    /// Data available from the solver, that can be updated by the user.
    pub solver_data: SolverDataMut<'a, P>,
    /// These are the values of each constraint at the end of the time step.
    pub constraint_values: &'a [Number],
    /// Objective value.
    pub objective_value: Number,
    /// Solve status. This enum reports the status of the last solve.
    pub status: SolveStatus,
}

/// Type defining the callback function for giving intermediate execution control to
/// the user. If set, it is called once per iteration, providing the user with some
/// information on the state of the optimization. This can be used to print some user-
/// defined output. It also gives the user a way to terminate the optimization
/// prematurely. If this method returns false, Ipopt will terminate the optimization.
pub type IntermediateCallback<P> = fn(&mut P, IntermediateCallbackData) -> bool;

/// Ipopt non-linear optimization problem solver. This structure is used to store data
/// needed to solve these problems using first and second order methods.
pub struct Ipopt<P: BasicProblem> {
    /// Internal (opaque) Ipopt problem representation.
    nlp_internal: ffi::CNLP_ProblemPtr,
    /// User specified interface defining the problem to be solved.
    nlp_interface: P,
    /// Intermediate callback.
    intermediate_callback: Option<IntermediateCallback<P>>,
    /// Number of primal variables.
    num_primal_variables: usize,
    /// Number of dual variables.
    num_dual_variables: usize,
}

/// Implement debug for Ipopt.
impl<P: BasicProblem + Debug> Debug for Ipopt<P> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f,
               "Ipopt {{ nlp_internal: {:?}, nlp_interface: {:?}, intermediate_callback: {:?}, num_primal_variables: {:?}, num_dual_variables: {:?} }}",
               self.nlp_internal,
               self.nlp_interface,
               if self.intermediate_callback.is_some() { "Some" } else { "None" },
               self.num_primal_variables,
               self.num_dual_variables)
    }
}

/// The only non-`Send` type in `Ipopt` is `nlp_internal`, which is a mutable raw pointer to an
/// underlying C struct. It is safe to implement `Send` for `Ipopt` here because it cannot be
/// copied or cloned.
unsafe impl<P: BasicProblem> Send for Ipopt<P> {}

impl<P: BasicProblem> Ipopt<P> {
    /// Common implementation for constructing an Ipopt struct. This involves some unsafe code that
    /// should be isolated for easier maintenance and debugging.
    fn new_impl(
        nlp_internal: ffi::CNLP_ProblemPtr,
        nlp: P,
        num_vars: usize,
        num_constraints: usize,
    ) -> Ipopt<P> {
        let mut ipopt = Ipopt {
            nlp_internal,
            nlp_interface: nlp,
            intermediate_callback: None,
            // These two will be updated every time sizes callback is called.
            num_primal_variables: num_vars,
            num_dual_variables: num_constraints,
        };

        // Initialize solution arrays so we can safely call solver_data and solver_data_mut without
        // addressing unallocated memory.
        unsafe {
            ffi::cnlp_init_solution(
                ipopt.nlp_internal,
                &mut ipopt as *mut Ipopt<P> as *mut std::ffi::c_void,
            );
        }

        ipopt
    }

    /// Create a new unconstrained non-linear problem.
    pub fn new_unconstrained(nlp: P) -> Result<Self, CreateError> {
        let num_vars = nlp.num_variables();

        // Ensure there is at least one variable given to optimize over.
        if num_vars < 1 {
            return Err(CreateError::NoOptimizationVariablesSpecified);
        }

        let mut nlp_internal: ffi::CNLP_ProblemPtr = ::std::ptr::null_mut();

        let create_error = CreateProblemStatus::new(unsafe {
            ffi::cnlp_create_problem(
                &mut nlp_internal as *mut ffi::CNLP_ProblemPtr,
                nlp.indexing_style() as Index,
                Some(Self::basic_sizes),
                Some(Self::basic_init),
                Some(Self::variable_only_bounds),
                Some(Self::eval_f),
                Some(Self::eval_g_none),
                Some(Self::eval_grad_f),
                Some(Self::eval_jac_g_none),
                Some(Self::eval_h_none),
                Some(Self::basic_scaling),
            )
        });

        if CreateProblemStatus::Success != create_error {
            return Err(create_error.into());
        }

        assert!(Self::set_ipopt_option(
            nlp_internal,
            "hessian_approximation",
            "limited-memory"
        ));

        Ok(Self::new_impl(nlp_internal, nlp, num_vars, 0))
    }

    /// Helper static function that can be used in the constructor.
    fn set_ipopt_option<'a, O>(nlp: ffi::CNLP_ProblemPtr, name: &str, option: O) -> bool
    where
        O: Into<IpoptOption<'a>>,
    {
        (unsafe {
            // Convert the input name string to a `char *` C type
            let name_cstr = CString::new(name).unwrap();

            // Match option to one of the three types of options Ipopt can receive.
            match option.into() {
                IpoptOption::Num(opt) => {
                    ffi::cnlp_add_num_option(nlp, name_cstr.as_ptr(), opt as Number)
                }
                IpoptOption::Str(opt) => {
                    // Convert option string to `char *`
                    let opt_cstr = CString::new(opt).unwrap();
                    ffi::cnlp_add_str_option(nlp, name_cstr.as_ptr(), opt_cstr.as_ptr())
                }
                IpoptOption::Int(opt) => {
                    ffi::cnlp_add_int_option(nlp, name_cstr.as_ptr(), opt as Int)
                }
            }
        } != 0) // converts Ipopt Bool to Rust bool
    }

    /// Set an Ipopt option.
    pub fn set_option<'a, O>(&mut self, name: &str, option: O) -> Option<&mut Self>
    where
        O: Into<IpoptOption<'a>>,
    {
        let success = Self::set_ipopt_option(self.nlp_internal, name, option);
        if success {
            Some(self)
        } else {
            None
        }
    }

    /// Set intermediate callback.
    pub fn set_intermediate_callback(&mut self, mb_cb: Option<IntermediateCallback<P>>)
    where
        P: BasicProblem,
    {
        self.intermediate_callback = mb_cb;

        unsafe {
            if mb_cb.is_some() {
                ffi::cnlp_set_intermediate_callback(self.nlp_internal, Some(Self::intermediate_cb));
            } else {
                ffi::cnlp_set_intermediate_callback(self.nlp_internal, None);
            }
        }
    }

    /// Solve non-linear problem.
    /// Return the solve status and the final value of the objective function.
    pub fn solve(&mut self) -> SolveResult<P> {
        let res = {
            let udata_ptr = self as *mut Ipopt<P>;
            unsafe { ffi::cnlp_solve(self.nlp_internal, udata_ptr as ffi::CNLP_UserDataPtr) }
        };

        let Ipopt {
            nlp_interface: ref mut problem,
            num_primal_variables,
            num_dual_variables,
            ..
        } = *self;

        SolveResult {
            solver_data: SolverDataMut {
                problem,
                solution: Solution::from_raw(res.data, num_primal_variables, num_dual_variables),
            },
            constraint_values: unsafe { slice::from_raw_parts(res.g, num_dual_variables) },
            objective_value: res.obj_val,
            status: SolveStatus::new(res.status),
        }
    }

    /// Get data for inspection and updating.
    #[allow(non_snake_case)]
    pub fn solver_data_mut(&mut self) -> SolverDataMut<P> {
        let Ipopt {
            nlp_interface: ref mut problem,
            nlp_internal,
            num_primal_variables,
            num_dual_variables,
            ..
        } = *self;

        let data = unsafe { ffi::cnlp_get_solver_data(nlp_internal) };

        SolverDataMut {
            problem,
            solution: Solution::from_raw(data, num_primal_variables, num_dual_variables),
        }
    }

    /// Get data for inspection from the internal solver.
    #[allow(non_snake_case)]
    pub fn solver_data(&self) -> SolverData<P> {
        let Ipopt {
            nlp_interface: ref problem,
            nlp_internal,
            num_primal_variables,
            num_dual_variables,
            ..
        } = *self;

        let data = unsafe { ffi::cnlp_get_solver_data(nlp_internal) };

        SolverData {
            problem,
            solution: Solution::from_raw(data, num_primal_variables, num_dual_variables),
        }
    }

    /**
     * Ipopt C API
     */

    /// Specify initial guess for variables and bounds multipliers. No constraints on basic
    /// problems.
    unsafe extern "C" fn basic_init(
        n: Index,
        init_x: Bool,
        x: *mut Number,
        init_z: Bool,
        z_l: *mut Number,
        z_u: *mut Number,
        m: Index,
        _init_lambda: Bool,
        _lambda: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        assert_eq!(m, 0);
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        if init_x != 0 {
            let x = slice::from_raw_parts_mut(x, n as usize);
            if !nlp.initial_point(x) {
                for i in 0..n as usize {
                    x[i] = 0.0;
                } // initialize to zero!
            }
        }
        if init_z != 0 {
            let z_l = slice::from_raw_parts_mut(z_l, n as usize);
            let z_u = slice::from_raw_parts_mut(z_u, n as usize);
            if !nlp.initial_bounds_multipliers(z_l, z_u) {
                for i in 0..n as usize {
                    z_l[i] = 0.0;
                    z_u[i] = 0.0;
                } // initialize to zero!
            }
        }
        true as Bool
    }

    /// Specify the number of elements needed to be allocated for the variables. No Hessian, and no
    /// constraints on basic problems.
    unsafe extern "C" fn basic_sizes(
        n: *mut Index,
        m: *mut Index,
        nnz_jac_g: *mut Index,
        nnz_h_lag: *mut Index,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let ipopt = &mut (*(user_data as *mut Ipopt<P>));

        ipopt.num_primal_variables = ipopt.nlp_interface.num_variables();
        ipopt.num_dual_variables = 0; // No constraints

        *n = ipopt.num_primal_variables as Index;
        *m = ipopt.num_dual_variables as Index;

        *nnz_jac_g = 0; // No constraints
        *nnz_h_lag = 0; // No Hessian
        true as Bool
    }

    /// Specify lower and upper bounds for variables. No constraints on basic problems.
    unsafe extern "C" fn variable_only_bounds(
        n: Index,
        x_l: *mut Number,
        x_u: *mut Number,
        m: Index,
        _g_l: *mut Number,
        _g_u: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        assert_eq!(m, 0);
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        nlp.bounds(
            slice::from_raw_parts_mut(x_l, n as usize),
            slice::from_raw_parts_mut(x_u, n as usize),
        ) as Bool
    }

    /// Evaluate the objective function.
    unsafe extern "C" fn eval_f(
        n: Index,
        x: *const Number,
        _new_x: Bool,
        obj_value: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        nlp.objective(slice::from_raw_parts(x, n as usize), &mut *obj_value) as Bool
    }

    /// Evaluate the objective gradient.
    unsafe extern "C" fn eval_grad_f(
        n: Index,
        x: *const Number,
        _new_x: Bool,
        grad_f: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        nlp.objective_grad(
            slice::from_raw_parts(x, n as usize),
            slice::from_raw_parts_mut(grad_f, n as usize),
        ) as Bool
    }

    /// Placeholder constraint function with no constraints.
    unsafe extern "C" fn eval_g_none(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _m: Index,
        _g: *mut Number,
        _user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        true as Bool
    }

    /// Placeholder constraint derivative function with no constraints.
    unsafe extern "C" fn eval_jac_g_none(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _m: Index,
        _nele_jac: Index,
        _irow: *mut Index,
        _jcol: *mut Index,
        _values: *mut Number,
        _user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        true as Bool
    }

    /// Placeholder hessian evaluation function.
    unsafe extern "C" fn eval_h_none(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _obj_factor: Number,
        _m: Index,
        _lambda: *const Number,
        _new_lambda: Bool,
        _nele_hess: Index,
        _irow: *mut Index,
        _jcol: *mut Index,
        _values: *mut Number,
        _user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        // From "Quasi-Newton Approximation of Second-Derivatives" in Ipopt docs:
        //  "If you are using the C or Fortran interface, you still need to implement [eval_h],
        //  but [it] should return false or IERR=1, respectively, and don't need to do
        //  anything else."
        false as Bool
    }

    /// Specify custom scaling parameters. This function is called by Ipopt when
    /// `nlp_scaling_method` is set to `user-scaling`.
    /// Basic problems have no constraint scaling.
    unsafe extern "C" fn basic_scaling(
        obj_scaling: *mut Number,
        use_x_scaling: *mut Bool,
        n: Index,
        x_scaling: *mut Number,
        use_g_scaling: *mut Bool,
        m: Index,
        _g_scaling: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        assert_eq!(m, 0);
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        *obj_scaling = nlp.objective_scaling();
        *use_x_scaling =
            nlp.variable_scaling(slice::from_raw_parts_mut(x_scaling, n as usize)) as Bool;
        *use_g_scaling = false as Bool;
        true as Bool
    }

    /// Intermediate callback.
    unsafe extern "C" fn intermediate_cb(
        alg_mod: ffi::CNLP_AlgorithmMode,
        iter_count: Index,
        obj_value: Number,
        inf_pr: Number,
        inf_du: Number,
        mu: Number,
        d_norm: Number,
        regularization_size: Number,
        alpha_du: Number,
        alpha_pr: Number,
        ls_trials: Index,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let ip = &mut (*(user_data as *mut Ipopt<P>));
        if let Some(callback) = ip.intermediate_callback {
            (callback)(
                &mut ip.nlp_interface,
                IntermediateCallbackData {
                    alg_mod: match alg_mod {
                        0 => AlgorithmMode::Regular,
                        _ => AlgorithmMode::RestorationPhase,
                    },
                    iter_count,
                    obj_value,
                    inf_pr,
                    inf_du,
                    mu,
                    d_norm,
                    regularization_size,
                    alpha_du,
                    alpha_pr,
                    ls_trials,
                },
            ) as Bool
        } else {
            true as Bool
        }
    }
}

impl<P: NewtonProblem> Ipopt<P> {
    /// Create a new second order problem, which will be solved using the Newton-Raphson method.
    pub fn new_newton(nlp: P) -> Result<Self, CreateError> {
        let num_vars = nlp.num_variables();

        // Ensure there is at least one variable given to optimize over.
        if num_vars < 1 {
            return Err(CreateError::NoOptimizationVariablesSpecified);
        }

        let mut nlp_internal: ffi::CNLP_ProblemPtr = ::std::ptr::null_mut();

        let create_error = CreateProblemStatus::new(unsafe {
            ffi::cnlp_create_problem(
                &mut nlp_internal as *mut ffi::CNLP_ProblemPtr,
                nlp.indexing_style() as Index,
                Some(Self::newton_sizes),
                Some(Self::basic_init),
                Some(Self::variable_only_bounds),
                Some(Self::eval_f),
                Some(Self::eval_g_none),
                Some(Self::eval_grad_f),
                Some(Self::eval_jac_g_none),
                Some(Self::eval_h),
                Some(Self::basic_scaling),
            )
        });

        if create_error != CreateProblemStatus::Success {
            return Err(create_error.into());
        }

        Ok(Self::new_impl(nlp_internal, nlp, num_vars, 0))
    }

    /**
     * Ipopt C API
     */

    /// Specify the number of elements needed to be allocated for the variables. No Hessian, and no
    /// constraints on basic problems.
    unsafe extern "C" fn newton_sizes(
        n: *mut Index,
        m: *mut Index,
        nnz_jac_g: *mut Index,
        nnz_h_lag: *mut Index,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        Self::basic_sizes(n, m, nnz_jac_g, nnz_h_lag, user_data);
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        *nnz_h_lag = nlp.num_hessian_non_zeros() as Index;
        true as Bool
    }

    /// Evaluate the hessian matrix.
    unsafe extern "C" fn eval_h(
        n: Index,
        x: *const Number,
        _new_x: Bool,
        obj_factor: Number,
        _m: Index,
        _lambda: *const Number,
        _new_lambda: Bool,
        nele_hess: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        if values.is_null() {
            /* return the structure. */
            nlp.hessian_indices(
                slice::from_raw_parts_mut(irow, nele_hess as usize),
                slice::from_raw_parts_mut(jcol, nele_hess as usize),
            ) as Bool
        } else {
            /* return the values. */
            let result = nlp.hessian_values(
                slice::from_raw_parts(x, n as usize),
                slice::from_raw_parts_mut(values, nele_hess as usize),
            ) as Bool;
            // This problem has no constraints so we can multiply each entry by the
            // objective factor.
            let start_idx = nlp.indexing_style() as isize;
            for i in start_idx..nele_hess as isize {
                *values.offset(i) *= obj_factor;
            }
            result
        }
    }
}

impl<P: ConstrainedProblem> Ipopt<P> {
    /// Create a new constrained non-linear problem.
    pub fn new(nlp: P) -> Result<Self, CreateError> {
        let num_vars = nlp.num_variables();

        // Ensure there is at least one variable given to optimize over.
        if num_vars < 1 {
            return Err(CreateError::NoOptimizationVariablesSpecified);
        }

        let num_constraints = nlp.num_constraints();
        let num_constraint_jac_nnz = nlp.num_constraint_jacobian_non_zeros();

        // It doesn't make sense to have constant constraints. Also if there is a non-trivial
        // constraint jacobian, there better be some constraints.
        // We check for these here explicitly to prevent hard-to-debug errors down the line.
        if (num_constraints > 0 && num_constraint_jac_nnz == 0)
            || (num_constraints == 0 && num_constraint_jac_nnz > 0)
        {
            return Err(CreateError::InvalidConstraintJacobian {
                num_constraints,
                num_constraint_jac_nnz,
            });
        }

        let mut nlp_internal: ffi::CNLP_ProblemPtr = ::std::ptr::null_mut();

        let create_error = CreateProblemStatus::new(unsafe {
            ffi::cnlp_create_problem(
                &mut nlp_internal as *mut ffi::CNLP_ProblemPtr,
                nlp.indexing_style() as Index,
                Some(Self::sizes),
                Some(Self::init),
                Some(Self::bounds),
                Some(Self::eval_f),
                Some(Self::eval_g),
                Some(Self::eval_grad_f),
                Some(Self::eval_jac_g),
                Some(Self::eval_full_h),
                Some(Self::scaling),
            )
        });

        if create_error != CreateProblemStatus::Success {
            return Err(create_error.into());
        }

        Ok(Self::new_impl(nlp_internal, nlp, num_vars, num_constraints))
    }

    /**
     * Ipopt C API
     */

    /// Specify initial guess for variables, bounds multipliers and constraint multipliers.
    unsafe extern "C" fn init(
        n: Index,
        init_x: Bool,
        x: *mut Number,
        init_z: Bool,
        z_l: *mut Number,
        z_u: *mut Number,
        m: Index,
        init_lambda: Bool,
        lambda: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        if init_x != 0 {
            let x = slice::from_raw_parts_mut(x, n as usize);
            if !nlp.initial_point(x) {
                for i in 0..n as usize {
                    x[i] = 0.0;
                } // initialize to zero!
            }
        }
        if init_z != 0 {
            let z_l = slice::from_raw_parts_mut(z_l, n as usize);
            let z_u = slice::from_raw_parts_mut(z_u, n as usize);
            if !nlp.initial_bounds_multipliers(z_l, z_u) {
                for i in 0..n as usize {
                    z_l[i] = 0.0;
                    z_u[i] = 0.0;
                } // initialize to zero!
            }
        }
        if init_lambda != 0 {
            let lambda = slice::from_raw_parts_mut(lambda, m as usize);
            if !nlp.initial_constraint_multipliers(lambda) {
                for i in 0..m as usize {
                    lambda[i] = 0.0;
                } // initialize to zero!
            }
        }
        true as Bool
    }

    /// Specify the number of elements needed to be allocated for various arrays.
    unsafe extern "C" fn sizes(
        n: *mut Index,
        m: *mut Index,
        nnz_jac_g: *mut Index,
        nnz_h_lag: *mut Index,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let ipopt = &mut (*(user_data as *mut Ipopt<P>));
        ipopt.num_primal_variables = ipopt.nlp_interface.num_variables();
        ipopt.num_dual_variables = ipopt.nlp_interface.num_constraints();

        *n = ipopt.num_primal_variables as Index;
        *m = ipopt.num_dual_variables as Index;

        *nnz_jac_g = ipopt.nlp_interface.num_constraint_jacobian_non_zeros() as Index;
        *nnz_h_lag = ipopt.nlp_interface.num_hessian_non_zeros() as Index;
        true as Bool
    }

    /// Specify lower and upper bounds for variables and the constraint function.
    unsafe extern "C" fn bounds(
        n: Index,
        x_l: *mut Number,
        x_u: *mut Number,
        m: Index,
        g_l: *mut Number,
        g_u: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        (nlp.bounds(
            slice::from_raw_parts_mut(x_l, n as usize),
            slice::from_raw_parts_mut(x_u, n as usize),
        ) && nlp.constraint_bounds(
            slice::from_raw_parts_mut(g_l, m as usize),
            slice::from_raw_parts_mut(g_u, m as usize),
        )) as Bool
    }

    /// Evaluate the constraint function.
    unsafe extern "C" fn eval_g(
        n: Index,
        x: *const Number,
        _new_x: Bool,
        m: Index,
        g: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        nlp.constraint(
            slice::from_raw_parts(x, n as usize),
            slice::from_raw_parts_mut(g, m as usize),
        ) as Bool
    }

    /// Evaluate the constraint Jacobian.
    unsafe extern "C" fn eval_jac_g(
        n: Index,
        x: *const Number,
        _new_x: Bool,
        _m: Index,
        nele_jac: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        if values.is_null() {
            /* return the structure of the Jacobian */
            nlp.constraint_jacobian_indices(
                slice::from_raw_parts_mut(irow, nele_jac as usize),
                slice::from_raw_parts_mut(jcol, nele_jac as usize),
            ) as Bool
        } else {
            /* return the values of the Jacobian of the constraints */
            nlp.constraint_jacobian_values(
                slice::from_raw_parts(x, n as usize),
                slice::from_raw_parts_mut(values, nele_jac as usize),
            ) as Bool
        }
    }

    /// Evaluate the hessian matrix. Compared to `eval_h` from `NewtonProblem`,
    /// this version includes the constraint hessian.
    unsafe extern "C" fn eval_full_h(
        n: Index,
        x: *const Number,
        _new_x: Bool,
        obj_factor: Number,
        m: Index,
        lambda: *const Number,
        _new_lambda: Bool,
        nele_hess: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        if values.is_null() {
            /* return the structure. */
            nlp.hessian_indices(
                slice::from_raw_parts_mut(irow, nele_hess as usize),
                slice::from_raw_parts_mut(jcol, nele_hess as usize),
            ) as Bool
        } else {
            /* return the values. */
            nlp.hessian_values(
                slice::from_raw_parts(x, n as usize),
                obj_factor,
                slice::from_raw_parts(lambda, m as usize),
                slice::from_raw_parts_mut(values, nele_hess as usize),
            ) as Bool
        }
    }

    /// Specify custom scaling parameters. This function is called by Ipopt when
    /// `nlp_scaling_method` is set to `user-scaling`.
    unsafe extern "C" fn scaling(
        obj_scaling: *mut Number,
        use_x_scaling: *mut Bool,
        n: Index,
        x_scaling: *mut Number,
        use_g_scaling: *mut Bool,
        m: Index,
        g_scaling: *mut Number,
        user_data: ffi::CNLP_UserDataPtr,
    ) -> Bool {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        *obj_scaling = nlp.objective_scaling();
        *use_x_scaling =
            nlp.variable_scaling(slice::from_raw_parts_mut(x_scaling, n as usize)) as Bool;
        *use_g_scaling =
            nlp.constraint_scaling(slice::from_raw_parts_mut(g_scaling, m as usize)) as Bool;
        true as Bool
    }
}

/// Free the memory allocated on the C side.
impl<P: BasicProblem> Drop for Ipopt<P> {
    fn drop(&mut self) {
        unsafe {
            ffi::cnlp_free_problem(self.nlp_internal);
        }
    }
}

/// Zero-based indexing (C Style) or one-based indexing (Fortran style).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum IndexingStyle {
    /// C-style array indexing starting from 0.
    CStyle = 0,
    /// Fortran-style array indexing starting from 1.
    FortranStyle = 1,
}

/// Program return status.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SolveStatus {
    /// Console Message: `EXIT: Optimal Solution Found.`
    ///
    /// This message indicates that IPOPT found a (locally) optimal point within the desired tolerances.
    SolveSucceeded,
    /// Console Message: `EXIT: Solved To Acceptable Level.`
    ///
    /// This indicates that the algorithm did not converge to the "desired" tolerances, but that
    /// it was able to obtain a point satisfying the "acceptable" tolerance level as specified by
    /// the [`acceptable_*`
    /// ](https://www.coin-or.org/Ipopt/documentation/node42.html#opt:acceptable_tol) options. This
    /// may happen if the desired tolerances are too small for the current problem.
    SolvedToAcceptableLevel,
    /// Console Message: `EXIT: Feasible point for square problem found.`
    ///
    /// This message is printed if the problem is "square" (i.e., it has as many equality
    /// constraints as free variables) and IPOPT found a feasible point.
    FeasiblePointFound,
    /// Console Message: `EXIT: Converged to a point of local infeasibility. Problem may be
    /// infeasible.`
    ///
    /// The restoration phase converged to a point that is a minimizer for the constraint violation
    /// (in the l1-norm), but is not feasible for the original problem. This indicates that
    /// the problem may be infeasible (or at least that the algorithm is stuck at a locally
    /// infeasible point). The returned point (the minimizer of the constraint violation) might
    /// help you to find which constraint is causing the problem. If you believe that the NLP is
    /// feasible, it might help to start the optimization from a different point.
    InfeasibleProblemDetected,
    /// Console Message: `EXIT: Search Direction is becoming Too Small.`
    ///
    /// This indicates that IPOPT is calculating very small step sizes and is making very little
    /// progress. This could happen if the problem has been solved to the best numerical accuracy
    /// possible given the current scaling.
    SearchDirectionBecomesTooSmall,
    /// Console Message: `EXIT: Iterates diverging; problem might be unbounded.`
    ///
    /// This message is printed if the max-norm of the iterates becomes larger than the value of
    /// the option [`diverging_iterates_tol`
    /// ](https://www.coin-or.org/Ipopt/documentation/node42.html#opt:diverging_iterates_tol).
    /// This can happen if the problem is unbounded below and the iterates are diverging.
    DivergingIterates,
    /// Console Message: `EXIT: Stopping optimization at current point as requested by user.`
    ///
    /// This message is printed if the user call-back method intermediate_callback returned false
    /// (see Section [3.3.4](https://www.coin-or.org/Ipopt/documentation/node23.html#sec:add_meth)).
    UserRequestedStop,
    /// Console Message: `EXIT: Maximum Number of Iterations Exceeded.`
    ///
    /// This indicates that IPOPT has exceeded the maximum number of iterations as specified by the
    /// option [`max_iter`](https://www.coin-or.org/Ipopt/documentation/node42.html#opt:max_iter).
    MaximumIterationsExceeded,
    /// Console Message: `EXIT: Maximum CPU time exceeded.`
    ///
    /// This indicates that IPOPT has exceeded the maximum number of CPU seconds as specified by
    /// the option
    /// [`max_cpu_time`](https://www.coin-or.org/Ipopt/documentation/node42.html#opt:max_cpu_time).
    MaximumCpuTimeExceeded,
    /// Console Message: `EXIT: Restoration Failed!`
    ///
    /// This indicates that the restoration phase failed to find a feasible point that was
    /// acceptable to the filter line search for the original problem. This could happen if the
    /// problem is highly degenerate, does not satisfy the constraint qualification, or if your NLP
    /// code provides incorrect derivative information.
    RestorationFailed,
    /// Console Output: `EXIT: Error in step computation (regularization becomes too large?)!`
    ///
    /// This messages is printed if IPOPT is unable to compute a search direction, despite several
    /// attempts to modify the iteration matrix. Usually, the value of the regularization parameter
    /// then becomes too large. One situation where this can happen is when values in the Hessian
    /// are invalid (NaN or Inf). You can check whether this is true by using the
    /// [`check_derivatives_for_naninf`
    /// ](https://www.coin-or.org/Ipopt/documentation/node44.html#opt:check_derivatives_for_naninf)
    /// option.
    ErrorInStepComputation,
    /// Console Message: (details about the particular error will be output to the console)
    ///
    /// This indicates that there was some problem specifying the options. See the specific message
    /// for details.
    InvalidOption,
    /// Console Message: `EXIT: Problem has too few degrees of freedom.`
    ///
    /// This indicates that your problem, as specified, has too few degrees of freedom. This can
    /// happen if you have too many equality constraints, or if you fix too many variables (IPOPT
    /// removes fixed variables by default, see also the [`fixed_variable_treatment`
    /// ](https://www.coin-or.org/Ipopt/documentation/node44.html#opt:fixed_variable_treatment)
    /// option).
    NotEnoughDegreesOfFreedom,
    /// Console Message: (no console message, this is a return code for the C and Fortran
    /// interfaces only.)
    ///
    /// This indicates that there was an exception of some sort when building the IpoptProblem
    /// structure in the C or Fortran interface. Likely there is an error in your model or the main
    /// routine.
    InvalidProblemDefinition,
    /// An invalid number like `NaN` was detected.
    InvalidNumberDetected,
    /// Console Message: (details about the particular error will be output to the console)
    ///
    /// This indicates that IPOPT has thrown an exception that does not have an internal return
    /// code. See the specific message for details.
    UnrecoverableException,
    /// Console Message: `Unknown Exception caught in Ipopt`
    ///
    /// An unknown exception was caught in IPOPT. This exception could have originated from your
    /// model or any linked in third party code.
    NonIpoptExceptionThrown,
    /// Console Message: `EXIT: Not enough memory.`
    ///
    /// An error occurred while trying to allocate memory. The problem may be too large for your
    /// current memory and swap configuration.
    InsufficientMemory,
    /// Console: `EXIT: INTERNAL ERROR: Unknown SolverReturn value - Notify IPOPT Authors.`
    ///
    /// An unknown internal error has occurred. Please notify the authors of IPOPT via the mailing
    /// list.
    InternalError,
    /// Unclassified error.
    UnknownError,
}

impl Display for SolveStatus {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match *self {
            SolveStatus::SolveSucceeded => write!(f, "
                Console Message: `EXIT: Optimal Solution Found.`\n\n\
                This message indicates that IPOPT found a (locally) optimal point within the desired \
                tolerances."),

            SolveStatus::SolvedToAcceptableLevel => write!(f, "
                Console Message: `EXIT: Solved To Acceptable Level.`\n\n\
                This indicates that the algorithm did not converge to the \"desired\" tolerances, but \
                that it was able to obtain a point satisfying the \"acceptable\" tolerance level as \
                specified by the `acceptable_*` options. This may happen if the desired tolerances \
                are too small for the current problem."),

            SolveStatus::FeasiblePointFound => write!(f, "
                Console Message: `EXIT: Feasible point for square problem found.`\n\n\
                This message is printed if the problem is \"square\" (i.e., it has as many equality \
                constraints as free variables) and IPOPT found a feasible point."),

            SolveStatus::InfeasibleProblemDetected => write!(f, "
                Console Message: `EXIT: Converged to a point of local infeasibility. Problem may be \
                infeasible.`\n\n\
                The restoration phase converged to a point that is a minimizer for the constraint \
                violation (in the l1-norm), but is not feasible for the original problem. This \
                indicates that the problem may be infeasible (or at least that the algorithm is stuck \
                at a locally infeasible point). The returned point (the minimizer of the constraint \
                violation) might help you to find which constraint is causing the problem. If you \
                believe that the NLP is feasible, it might help to start the optimization from a \
                different point."),

            SolveStatus::SearchDirectionBecomesTooSmall => write!(f, "
                Console Message: `EXIT: Search Direction is becoming Too Small.`\n\n\
                This indicates that IPOPT is calculating very small step sizes and is making very \
                little progress. This could happen if the problem has been solved to the best numerical \
                accuracy possible given the current scaling."),

            SolveStatus::DivergingIterates => write!(f, "
                Console Message: `EXIT: Iterates diverging; problem might be unbounded.`\n\n\
                This message is printed if the max-norm of the iterates becomes larger than the value \
                of the option `diverging_iterates_tol`. This can happen if the problem is unbounded \
                below and the iterates are diverging."),

            SolveStatus::UserRequestedStop => write!(f, "
                Console Message: `EXIT: Stopping optimization at current point as requested by \
                user.`\n\n\
                This message is printed if the user call-back method intermediate_callback returned \
                false."),

            SolveStatus::MaximumIterationsExceeded => write!(f, "
                Console Message: `EXIT: Maximum Number of Iterations Exceeded.`\n\n\
                This indicates that IPOPT has exceeded the maximum number of iterations as specified by \
                the option `max_iter`."),

            SolveStatus::MaximumCpuTimeExceeded => write!(f, "
                Console Message: `EXIT: Maximum CPU time exceeded.`\n\n\
                This indicates that IPOPT has exceeded the maximum number of CPU seconds as specified \
                by the option `max_cpu_time`."),

            SolveStatus::RestorationFailed => write!(f, "
                Console Message: `EXIT: Restoration Failed!`\n\n\
                This indicates that the restoration phase failed to find a feasible point that was \
                acceptable to the filter line search for the original problem. This could happen if the \
                problem is highly degenerate, does not satisfy the constraint qualification, or if \
                your NLP code provides incorrect derivative information."),

            SolveStatus::ErrorInStepComputation => write!(f, "
                Console Output: `EXIT: Error in step computation (regularization becomes too large?)!`\n\n\
                This messages is printed if IPOPT is unable to compute a search direction, despite \
                several attempts to modify the iteration matrix. Usually, the value of the \
                regularization parameter then becomes too large. One situation where this can happen is \
                when values in the Hessian are invalid (NaN or Inf). You can check whether this is \
                true by using the `check_derivatives_for_naninf` option."),

            SolveStatus::InvalidOption => write!(f, "
                Console Message: (details about the particular error will be output to the console)\n\n\
                This indicates that there was some problem specifying the options. See the specific \
                message for details."),

            SolveStatus::NotEnoughDegreesOfFreedom => write!(f, "
                Console Message: `EXIT: Problem has too few degrees of freedom.`\n\n\
                This indicates that your problem, as specified, has too few degrees of freedom. This \
                can happen if you have too many equality constraints, or if you fix too many variables \
                (IPOPT removes fixed variables by default, see also the `fixed_variable_treatment` \
                option)."),

            SolveStatus::InvalidProblemDefinition => write!(f, "
                Console Message: (no console message, this is a return code for the C and Fortran \
                interfaces only.)\n\n\
                This indicates that there was an exception of some sort when building the \
                IpoptProblem structure in the C or Fortran interface. Likely there is an error in \
                your model or the main routine."),

            SolveStatus::InvalidNumberDetected => write!(f, "An invalid number like `NaN` was detected."),

            SolveStatus::UnrecoverableException => write!(f, "
                Console Message: (details about the particular error will be output to the \
                console)\n\n\
                This indicates that IPOPT has thrown an exception that does not have an internal \
                return code. See the specific message for details."),

            SolveStatus::NonIpoptExceptionThrown => write!(f, "
                Console Message: `Unknown Exception caught in Ipopt`\n\n\
                An unknown exception was caught in IPOPT. This exception could have originated from \
                your model or any linked in third party code."),

            SolveStatus::InsufficientMemory => write!(f, "
                Console Message: `EXIT: Not enough memory.`\n\n\
                An error occurred while trying to allocate memory. The problem may be too large for \
                your current memory and swap configuration."),

            SolveStatus::InternalError => write!(f, "
                Console: `EXIT: INTERNAL ERROR: Unknown SolverReturn value - Notify IPOPT Authors.`\n\n\
                An unknown internal error has occurred. Please notify the authors of IPOPT via the \
                mailing list."),

            SolveStatus::UnknownError => write!(f, "Unclassified error."),
        }
    }
}

#[allow(non_snake_case)]
impl SolveStatus {
    fn new(status: ffi::CNLP_ApplicationReturnStatus) -> Self {
        use crate::SolveStatus as RS;
        match status {
            ffi::CNLP_ApplicationReturnStatus_CNLP_SOLVE_SUCCEEDED => RS::SolveSucceeded,
            ffi::CNLP_ApplicationReturnStatus_CNLP_SOLVED_TO_ACCEPTABLE_LEVEL => {
                RS::SolvedToAcceptableLevel
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_INFEASIBLE_PROBLEM_DETECTED => {
                RS::InfeasibleProblemDetected
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_SEARCH_DIRECTION_BECOMES_TOO_SMALL => {
                RS::SearchDirectionBecomesTooSmall
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_DIVERGING_ITERATES => RS::DivergingIterates,
            ffi::CNLP_ApplicationReturnStatus_CNLP_USER_REQUESTED_STOP => RS::UserRequestedStop,
            ffi::CNLP_ApplicationReturnStatus_CNLP_FEASIBLE_POINT_FOUND => RS::FeasiblePointFound,
            ffi::CNLP_ApplicationReturnStatus_CNLP_MAXIMUM_ITERATIONS_EXCEEDED => {
                RS::MaximumIterationsExceeded
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_RESTORATION_FAILED => RS::RestorationFailed,
            ffi::CNLP_ApplicationReturnStatus_CNLP_ERROR_IN_STEP_COMPUTATION => {
                RS::ErrorInStepComputation
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_MAXIMUM_CPUTIME_EXCEEDED => {
                RS::MaximumCpuTimeExceeded
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_NOT_ENOUGH_DEGREES_OF_FREEDOM => {
                RS::NotEnoughDegreesOfFreedom
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_INVALID_PROBLEM_DEFINITION => {
                RS::InvalidProblemDefinition
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_INVALID_OPTION => RS::InvalidOption,
            ffi::CNLP_ApplicationReturnStatus_CNLP_INVALID_NUMBER_DETECTED => {
                RS::InvalidNumberDetected
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_UNRECOVERABLE_EXCEPTION => {
                RS::UnrecoverableException
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_NONIPOPT_EXCEPTION_THROWN => {
                RS::NonIpoptExceptionThrown
            }
            ffi::CNLP_ApplicationReturnStatus_CNLP_INSUFFICIENT_MEMORY => RS::InsufficientMemory,
            ffi::CNLP_ApplicationReturnStatus_CNLP_INTERNAL_ERROR => RS::InternalError,
            _ => RS::UnknownError,
        }
    }
}

/// Problem create error type. This type is higher level than `CreateProblemStatus` and it captures
/// inconsistencies with the input before even calling `CreateIpoptProblem` internally adding
/// safety to this wrapper.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CreateError {
    /// No optimization variables were provided.
    NoOptimizationVariablesSpecified,
    /// The number of Jacobian elements is non-zero, yet no constraints were provided or
    /// the number of constraints is non-zero, yet no Jacobian elements were provided.
    InvalidConstraintJacobian {
        /// Number of constraints set in the problem.
        num_constraints: usize,
        /// Number of constraint Jacobian entries specified for the problem.
        num_constraint_jac_nnz: usize,
    },
    /// Unexpected error occurred: None of the above. This is likely an internal bug.
    Unknown,
}

impl Display for CreateError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match *self {
            CreateError::NoOptimizationVariablesSpecified => {
                write!(f, "No optimization variables were provided.")
            }
            CreateError::InvalidConstraintJacobian {
                num_constraints,
                num_constraint_jac_nnz,
            } => write!(
                f,
                "The number of constraint Jacobian elements ({}) is inconsistent with the \
                 number of constraints ({}).",
                num_constraint_jac_nnz, num_constraints
            ),
            CreateError::Unknown => write!(
                f,
                "Unexpected error occurred. This is likely an internal bug."
            ),
        }
    }
}

impl std::error::Error for CreateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<CreateProblemStatus> for CreateError {
    fn from(_s: CreateProblemStatus) -> CreateError {
        CreateError::Unknown
    }
}

/// Internal program create return status.
#[derive(Copy, Clone, Debug, PartialEq)]
enum CreateProblemStatus {
    /// Program creation was successful. This variant should never be returned, instead a
    /// successfully built instance is returned in a `Result` struct.
    Success,
    /// Missing callback for determining sizes of data arrays .
    MissingSizes,
    /// The initial guess callback is missing.
    MissingInitialGuess,
    /// Missing callback for evaluating variable and constraint bounds.
    MissingBounds,
    /// Missing callback for evaluating the objective: `eval_f`.
    MissingEvalF,
    /// Missing callback for evaluating the gradient of the objective: `eval_grad_f`.
    MissingEvalGradF,
    /// Inconsistent problem definition.
    InvalidProblemDefinition,
    /// Unexpected error occurred: None of the above. This is likely an internal bug.
    UnknownError,
}

#[allow(non_snake_case)]
impl CreateProblemStatus {
    fn new(status: ffi::CNLP_CreateProblemStatus) -> Self {
        use crate::CreateProblemStatus as RS;
        match status {
            ffi::CNLP_CreateProblemStatus_CNLP_SUCCESS => RS::Success,
            ffi::CNLP_CreateProblemStatus_CNLP_MISSING_INITIAL_GUESS => RS::MissingInitialGuess,
            ffi::CNLP_CreateProblemStatus_CNLP_MISSING_BOUNDS => RS::MissingBounds,
            ffi::CNLP_CreateProblemStatus_CNLP_MISSING_EVAL_F => RS::MissingEvalF,
            ffi::CNLP_CreateProblemStatus_CNLP_MISSING_EVAL_GRAD_F => RS::MissingEvalGradF,
            ffi::CNLP_CreateProblemStatus_CNLP_INVALID_PROBLEM_DEFINITION_ON_CREATE => {
                RS::InvalidProblemDefinition
            }
            ffi::CNLP_CreateProblemStatus_CNLP_UNRECOVERABLE_EXCEPTION_ON_CREATE => {
                RS::UnknownError
            }
            _ => RS::UnknownError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct NlpUnconstrained {
        num_vars: usize,
        init_point: Vec<Number>,
        lower: Vec<Number>,
        upper: Vec<Number>,
    }

    impl BasicProblem for NlpUnconstrained {
        fn num_variables(&self) -> usize {
            self.num_vars
        }
        fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
            x_l.copy_from_slice(&self.lower);
            x_u.copy_from_slice(&self.upper);
            true
        }
        fn initial_point(&self, x: &mut [Number]) -> bool {
            x.copy_from_slice(&self.init_point);
            true
        }
        fn objective(&self, _: &[Number], _: &mut Number) -> bool {
            true
        }
        fn objective_grad(&self, _: &[Number], _: &mut [Number]) -> bool {
            true
        }
    }

    /// Test validation of new unconstrained ipopt problems.
    #[test]
    fn invalid_construction_unconstrained_test() {
        // Initialize a valid nlp.
        let nlp = NlpUnconstrained {
            num_vars: 2,
            init_point: vec![0.0, 0.0],
            lower: vec![-1e20; 2],
            upper: vec![1e20; 2],
        };

        assert!(Ipopt::new_unconstrained(nlp.clone()).is_ok());

        // Invalid number of variables
        let nlp4 = NlpUnconstrained {
            num_vars: 0,
            ..nlp.clone()
        };
        assert_eq!(
            Ipopt::new_unconstrained(nlp4).unwrap_err(),
            CreateError::NoOptimizationVariablesSpecified
        );
    }

    #[derive(Debug, Clone)]
    struct NlpConstrained {
        num_vars: usize,
        num_constraints: usize,
        num_constraint_jac_nnz: usize,
        num_hess_nnz: usize,
        constraint_lower: Vec<Number>,
        constraint_upper: Vec<Number>,
        init_point: Vec<Number>,
        lower: Vec<Number>,
        upper: Vec<Number>,
    }

    impl BasicProblem for NlpConstrained {
        fn num_variables(&self) -> usize {
            self.num_vars
        }
        fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
            x_l.copy_from_slice(&self.lower);
            x_u.copy_from_slice(&self.upper);
            true
        }
        fn initial_point(&self, x: &mut [Number]) -> bool {
            x.copy_from_slice(&self.init_point.clone());
            true
        }
        fn objective(&self, _: &[Number], _: &mut Number) -> bool {
            true
        }
        fn objective_grad(&self, _: &[Number], _: &mut [Number]) -> bool {
            true
        }
    }

    impl ConstrainedProblem for NlpConstrained {
        fn num_constraints(&self) -> usize {
            self.num_constraints
        }
        fn num_constraint_jacobian_non_zeros(&self) -> usize {
            self.num_constraint_jac_nnz
        }

        fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
            g_l.copy_from_slice(&self.constraint_lower);
            g_u.copy_from_slice(&self.constraint_upper);
            true
        }
        fn constraint(&self, _: &[Number], _: &mut [Number]) -> bool {
            true
        }
        fn constraint_jacobian_indices(&self, _: &mut [Index], _: &mut [Index]) -> bool {
            true
        }
        fn constraint_jacobian_values(&self, _: &[Number], _: &mut [Number]) -> bool {
            true
        }

        // Hessian Implementation
        fn num_hessian_non_zeros(&self) -> usize {
            self.num_hess_nnz
        }
        fn hessian_indices(&self, _: &mut [Index], _: &mut [Index]) -> bool {
            true
        }
        fn hessian_values(&self, _: &[Number], _: Number, _: &[Number], _: &mut [Number]) -> bool {
            true
        }
    }

    /// Test validation of new constrained ipopt problems.
    #[test]
    fn invalid_construction_constrained_test() {
        // Initialize a valid nlp.
        let nlp = NlpConstrained {
            num_vars: 4,
            num_constraints: 2,
            num_constraint_jac_nnz: 8,
            num_hess_nnz: 10,
            constraint_lower: vec![25.0, 40.0],
            constraint_upper: vec![2.0e19, 40.0],
            init_point: vec![1.0, 5.0, 5.0, 1.0],
            lower: vec![1.0; 4],
            upper: vec![5.0; 4],
        };

        assert!(Ipopt::new(nlp.clone()).is_ok());

        // Invalid number of variables
        let nlp2 = NlpConstrained {
            num_vars: 0,
            ..nlp.clone()
        };
        assert_eq!(
            Ipopt::new(nlp2).unwrap_err(),
            CreateError::NoOptimizationVariablesSpecified
        );

        // Invalid constraint jacobian
        let nlp3 = NlpConstrained {
            num_constraint_jac_nnz: 0,
            ..nlp.clone()
        };
        assert_eq!(
            Ipopt::new(nlp3).unwrap_err(),
            CreateError::InvalidConstraintJacobian {
                num_constraints: 2,
                num_constraint_jac_nnz: 0
            }
        );

        let nlp4 = NlpConstrained {
            num_constraints: 0,
            constraint_lower: vec![],
            constraint_upper: vec![],
            ..nlp.clone()
        };
        assert_eq!(
            Ipopt::new(nlp4).unwrap_err(),
            CreateError::InvalidConstraintJacobian {
                num_constraints: 0,
                num_constraint_jac_nnz: 8
            }
        );
    }

    /// Test validity of solver data before the first solve.
    /// Here we ensure that the necessary arrays are allocated at the time of creation.
    #[test]
    fn no_solve_validity_test() {
        // Initialize a valid nlp.
        let nlp = NlpConstrained {
            num_vars: 4,
            num_constraints: 2,
            num_constraint_jac_nnz: 8,
            num_hess_nnz: 10,
            constraint_lower: vec![25.0, 40.0],
            constraint_upper: vec![2.0e19, 40.0],
            init_point: vec![1.0, 5.0, 5.0, 1.0],
            lower: vec![1.0; 4],
            upper: vec![5.0; 4],
        };

        let solver = Ipopt::new(nlp.clone()).expect("Failed to create Ipopt solver");
        let SolverData {
            solution:
                Solution {
                    primal_variables,
                    constraint_multipliers,
                    lower_bound_multipliers,
                    upper_bound_multipliers,
                },
            ..
        } = solver.solver_data();

        // We expect that the solver data is initialized to what is provided by the initial_*
        // functions.  Although they will be called again during the actual solve, the data
        // returned by solver_data must always be valid.
        assert_eq!(primal_variables, nlp.init_point.as_slice());
        assert_eq!(constraint_multipliers, vec![0.0; 2].as_slice());
        assert_eq!(lower_bound_multipliers, vec![0.0; 4].as_slice());
        assert_eq!(upper_bound_multipliers, vec![0.0; 4].as_slice());
    }
}
