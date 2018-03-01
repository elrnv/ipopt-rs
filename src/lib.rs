extern crate ipopt_sys as ffi;

use ffi::{Bool, Int};
use std::ffi::CString;
use std::slice;

pub type Number = f64; // Same as ffi::Number
pub type Index = i32;  // Same as ffi::Index

/// The non-linear problem to be solved by Ipopt. This trait specifies all the
/// information needed to construct the unconstrained optimization problem (although the
/// variables are allowed to be bounded).
/// In the callbacks within, `x` is the independent variable and must be the same size
/// as returned by `num_variables`.
/// Each of the callbacks required during interior point iterations are allowed to fail.
/// In case of failure to produce values, simply return `false` where applicable.
/// This feature could be used to tell Ipopt to try smaller perturbations for `x` for
/// instance.
pub trait BasicProblem {
    /// Specify the indexing style used for arrays in this problem.
    /// (Default is zero-based)
    fn indexing_style(&self) -> IndexingStyle { IndexingStyle::CStyle }
    /// Total number of variables of the non-linear problem.
    fn num_variables(&self) -> usize;

    /// Specify the pair of variable bounds `(lower, upper)`.
    /// The returned `Vec`s must have the same size as `num_variables`.
    fn bounds(&self) -> (Vec<Number>, Vec<Number>);

    /// Construct the initial guess for Ipopt to start with.
    /// The returned `Vec` must have the same size as `num_variables`.
    fn initial_point(&self) -> Vec<Number>;

    /// Objective function. This is the function being minimized.
    /// This function is internally called by Ipopt callback `eval_f`.
    fn objective(&mut self, x: &[Number], obj: &mut Number) -> bool;
    /// Gradient of the objective function.
    /// This function is internally called by Ipopt callback `eval_grad_f`.
    fn objective_grad(&mut self, x: &[Number], grad_f: &mut [Number]) -> bool;
}

/// An extension to the `BasicProblem` trait that enables full Newton iterations in
/// Ipopt. If this trait is NOT implemented by your problem, Ipopt will be set to perform
/// [Quasi-Newton Approximation](https://www.coin-or.org/Ipopt/documentation/node31.html)
/// for second derivatives.
/// This interface asks for the Hessian matrix in sparse triplet form.
pub trait NewtonProblem : BasicProblem {
    /// Number of non-zeros in the Hessian matrix. This includes the constraint hessian
    /// if one is provided.
    fn num_hessian_non_zeros(&self) -> usize;
    /// Hessian indices. These are the row and column indices of the non-zeros
    /// in the sparse representation of the matrix.
    /// This is a symmetric matrix, fill the lower left triangular half only.
    /// If your problem is constrained (i.e. you are ultimately implementing
    /// `ConstrainedProblem`), ensure that you provide coordinates for non-zeros of the
    /// constraint hessian as well.
    /// This function is internally called by Ipopt callback `eval_h`.
    fn hessian_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Objective Hessian values. Each value must correspond to the `row` and `column` as
    /// specified in `hessian_indices`.
    /// This function is internally called by Ipopt callback `eval_h` and each value is
    /// premultiplied by `Ipopt`'s `obj_factor` as necessary.
    fn hessian_values(&mut self, x: &[Number], vals: &mut [Number]) -> bool;
}

/// Extends the `BasicProblem` trait to enable equality and inequality constraints.
/// Equality constraints are enforce by setting the lower and upper bounds for the
/// constraint to the same value.
/// This type of problem is the target use case for Ipopt.
/// NOTE: Although its possible to run Quasi-Newton iterations on a constrained problem,
/// it doesn't perform well in general, which is the reason why you must also provide the
/// Hessian callbacks.  However, you may still enable L-BFGS explicitly by setting the
/// "hessian_approximation" Ipopt option to "limited-memory", in which case you should
/// simply return `false` in `hessian_indices` and `hessian_values`.
pub trait ConstrainedProblem : BasicProblem {
    /// Number of equality and inequality constraints.
    fn num_constraints(&self) -> usize;
    /// Number of non-zeros in the constraint jacobian.
    fn num_constraint_jac_non_zeros(&self) -> usize;
    /// Constraint function. This gives the value of each constraint.
    /// The output slice `g` is guaranteed to be the same size as `num_constraints`.
    /// This function is internally called by Ipopt callback `eval_g`.
    fn constraint(&mut self, x: &[Number], g: &mut [Number]) -> bool;
    /// Specify the pair of bounds `(lower, upper)` on the value of the constraint
    /// function.
    /// The returned `Vec`s must each have the same size as `num_constraints`.
    fn constraint_bounds(&self) -> (Vec<Number>, Vec<Number>);
    /// Constraint jacobian indices. These are the row and column indices of the
    /// non-zeros in the sparse representation of the matrix.
    /// This function is internally called by Ipopt callback `eval_jac_g`.
    fn constraint_jac_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Constraint jacobian values. Each value must correspond to the `row` and
    /// `column` as specified in `constraint_jac_indices`.
    /// This function is internally called by Ipopt callback `eval_jac_g`.
    fn constraint_jac_values(&mut self, x: &[Number], vals: &mut [Number]) -> bool;
    /// Number of non-zeros in the Hessian matrix. This includes the constraint hessian.
    fn num_hessian_non_zeros(&self) -> usize;
    /// Hessian indices. These are the row and column indices of the non-zeros
    /// in the sparse representation of the matrix.
    /// This should be a symmetric matrix, fill the lower left triangular half only.
    /// Ensure that you provide coordinates for non-zeros of the
    /// objective and constraint hessians.
    /// This function is internally called by Ipopt callback `eval_h`.
    fn hessian_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Hessian values. Each value must correspond to the `row` and `column` as
    /// specified in `hessian_indices`.
    /// Write the objective hessian values multiplied by `obj_factor` and constraint
    /// hessian values multipled by the corresponding values in `lambda` (the Lagrange
    /// multiplier).
    /// This function is internally called by Ipopt callback `eval_h`.
    fn hessian_values(&mut self,
                      x: &[Number],
                      obj_factor: Number,
                      lambda: &[Number],
                      vals: &mut [Number]) -> bool;
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

/// Type defining the callback function for giving intermediate execution control to
/// the user. If set, it is called once per iteration, providing the user with some
/// information on the state of the optimization. This can be used to print some user-
/// defined output. It also gives the user a way to terminate the optimization
/// prematurely. If this method returns false, Ipopt will terminate the optimization.
pub type IntermediateCallback<P> = 
    fn(&mut P, Index,Index,
       Number,Number,Number,
       Number,Number,Number,
       Number,Number,Index) -> bool;

/// Ipopt non-linear optimization problem solver. This structure is used to store data
/// needed to solve these problems using first and second order methods.
pub struct Ipopt<P: BasicProblem> {
    /// Internal (opaque) Ipopt problem representation.
    nlp_internal: ffi::IpoptProblem,
    /// User specified interface defining the problem to be solved.
    nlp_interface: P,
    /// Constraint multipliers.
    mult_g: Vec<Number>,
    /// Variable lower bound multipliers.
    mult_x_l: Vec<Number>,
    /// Variable upper bound multipliers.
    mult_x_u: Vec<Number>,
    /// Vector of variables. This stores the initial guess and the solution.
    x: Vec<Number>,
    /// Intermediate callback.
    intermediate_callback: Option<IntermediateCallback<P>>,
}


impl<P: BasicProblem> Ipopt<P> {
    /// Create a new unconstrained non-linear problem.
    pub fn new_unconstrained(nlp: P) -> Self {
        let (mut x_l, mut x_u) = nlp.bounds();

        let num_vars = nlp.num_variables();

        let nlp_internal = unsafe {
            ffi::CreateIpoptProblem(
                num_vars as Index,
                x_l.as_mut_ptr(),
                x_u.as_mut_ptr(),
                0, // no constraints
                ::std::ptr::null_mut(),
                ::std::ptr::null_mut(),
                0, // no non-zeros in constraint jacobian
                0, // no hessian
                nlp.indexing_style() as Index,
                Some(Self::eval_f),
                Some(Self::eval_g_none),
                Some(Self::eval_grad_f),
                Some(Self::eval_jac_g_none),
                Some(Self::eval_h_none))
        };

        let mut mult_x_l = Vec::with_capacity(num_vars);
        mult_x_l.resize(num_vars, 0.0);
        let mut mult_x_u = Vec::with_capacity(num_vars);
        mult_x_u.resize(num_vars, 0.0);
        let x = nlp.initial_point();

        assert!(Self::set_ipopt_option(nlp_internal,
                                       "hessian_approximation",
                                       "limited-memory"));
        Ipopt {
            nlp_internal,
            nlp_interface: nlp,
            mult_g: Vec::new(),
            mult_x_l,
            mult_x_u,
            x,
            intermediate_callback: None,
        }
    }

    /// Get an immutable reference to the provided NLP object.
    pub fn problem(&self) -> &P {
        &self.nlp_interface
    }
    
    /// Get a mutable reference to the provided NLP object.
    pub fn problem_mut(&mut self) -> &mut P {
        &mut self.nlp_interface
    }

    /// Helper static function that can be used in the constructor.
    fn set_ipopt_option<'a, O>(nlp: ffi::IpoptProblem, name: &str, option: O) -> bool
        where O: Into<IpoptOption<'a>>
    {
        let success = unsafe {
            // Convert the input name string to a `char *` C type
            let mut name_cstr = CString::new(name).unwrap();
            let name_cstr = (&mut name_cstr).as_ptr() as *mut i8; // this is `char *`
            // Match option to one of the three types of options Ipopt can receive.
            match option.into() {
                IpoptOption::Num(opt) =>
                    ffi::AddIpoptNumOption(nlp, name_cstr, opt as Number),
                IpoptOption::Str(opt) => {
                    // Convert option string to `char *`
                    let mut opt_cstr = CString::new(opt).unwrap();
                    let opt_cstr = (&mut opt_cstr).as_ptr() as *mut i8;
                    ffi::AddIpoptStrOption(nlp, name_cstr, opt_cstr)
                },
                IpoptOption::Int(opt) =>
                    ffi::AddIpoptIntOption(nlp, name_cstr, opt as Int),
            }
        } != 0; // converts Ipopt Bool to Rust bool
        success
    }

    /// Set an Ipopt option.
    pub fn set_option<'a, O>(&mut self, name: &str, option: O) -> Option<&mut Self>
        where O: Into<IpoptOption<'a>>
    {
        let success = Self::set_ipopt_option(self.nlp_internal, name, option);
        if success { Some(self) } else { None }
    }

    /// Set intermediate callback.
    pub fn set_intermediate_callback(&mut self, mb_cb: Option<IntermediateCallback<P>>)
        where P: BasicProblem,
    {
        self.intermediate_callback = mb_cb;

        unsafe {
            if let Some(_) = mb_cb {
                ffi::SetIntermediateCallback(self.nlp_internal, Some(Self::intermediate_cb));
            } else {
                ffi::SetIntermediateCallback(self.nlp_internal, None);
            }
        }
    }

    /// Solve non-linear problem.
    /// Return the solve status and the final value of the objective function.
    pub fn solve(&mut self) -> (ReturnStatus, Number) {
        let mut objective_value = 0.0;
        let udata_ptr = self as *mut Ipopt<P>;
        let status = ReturnStatus::new( unsafe {
            ffi::IpoptSolve(
                self.nlp_internal,
                self.x.as_mut_ptr(),
                ::std::ptr::null_mut(),
                &mut objective_value as *mut Number,
                if self.mult_g.is_empty() {
                    ::std::ptr::null_mut()
                } else {
                    self.mult_g.as_mut_ptr() },
                self.mult_x_l.as_mut_ptr(),
                self.mult_x_u.as_mut_ptr(),
                udata_ptr as ffi::UserDataPtr)
        });

        (status, objective_value)
    }

    /// Return the multipliers that enforce the variable bounds.
    pub fn bound_multipliers(&self) -> (&[Number], &[Number]) {
        (self.mult_x_l.as_slice(), self.mult_x_u.as_slice())
    }

    /// Return the multipliers that enforce the variable bounds.
    pub fn solution(&self) -> &[Number] {
        self.x.as_slice()
    }

    /**
     * Ipopt C API
     */

    /// Evaluate the objective function.
    unsafe extern "C" fn eval_f(
        n: Index,
        x: *mut Number,
        _new_x: Bool,
        obj_value: *mut Number,
        user_data: ffi::UserDataPtr) -> Bool
    {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        nlp.objective(slice::from_raw_parts(x, n as usize), &mut *obj_value) as Bool
    }

    /// Evaluate the objective gradient.
    unsafe extern "C" fn eval_grad_f(
        n: Index,
        x: *mut Number,
        _new_x: Bool,
        grad_f: *mut Number,
        user_data: ffi::UserDataPtr) -> Bool
    {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        nlp.objective_grad(slice::from_raw_parts(x, n as usize),
                           slice::from_raw_parts_mut(grad_f, n as usize)) as Bool
    }

    /// Placeholder constraint function with no constraints.
    unsafe extern "C" fn eval_g_none(
        _n: Index,
        _x: *mut Number,
        _new_x: Bool,
        _m: Index,
        _g: *mut Number,
        _user_data: ffi::UserDataPtr) -> Bool
    {
        true as Bool
    }

    /// Placeholder constraint derivative function with no constraints.
    unsafe extern "C" fn eval_jac_g_none(
        _n: Index,
        _x: *mut Number,
        _new_x: Bool,
        _m: Index,
        _nele_jac: Index,
        _irow: *mut Index,
        _jcol: *mut Index,
        _values: *mut Number,
        _user_data: ffi::UserDataPtr) -> Bool
    {
        true as Bool
    }

    /// Placeholder hessian evaluation function.
    unsafe extern "C" fn eval_h_none(
        _n: Index,
        _x: *mut Number,
        _new_x: Bool,
        _obj_factor: Number,
        _m: Index,
        _lambda: *mut Number,
        _new_lambda: Bool,
        _nele_hess: Index,
        _irow: *mut Index,
        _jcol: *mut Index,
        _values: *mut Number,
        _user_data: ffi::UserDataPtr) -> Bool
    {
        // From "Quasi-Newton Approximation of Second-Derivatives" in Ipopt docs:
        //  "If you are using the C or Fortran interface, you still need to implement [eval_h],
        //  but [it] should return false or IERR=1, respectively, and don't need to do
        //  anything else."
        false as Bool
    }

    /// Intermediate callback.
    unsafe extern "C" fn intermediate_cb(
        alg_mod: Index,
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
        user_data: ffi::UserDataPtr) -> Bool
    {
        let ip = &mut (*(user_data as *mut Ipopt<P>));
        if let Some(callback) = ip.intermediate_callback {
            (callback)(
                &mut ip.nlp_interface,
                alg_mod, iter_count, obj_value,
                inf_pr, inf_du, mu, d_norm,
                regularization_size, alpha_du, alpha_pr, ls_trials) as Bool
        } else {
            true as Bool
        }
    }
}

impl<P: NewtonProblem> Ipopt<P> {
    /// Create a new newton problem.
    pub fn new_newton(nlp: P) -> Self {
        let (mut x_l, mut x_u) = nlp.bounds();

        let num_vars = nlp.num_variables();

        let nlp_internal = unsafe {
            ffi::CreateIpoptProblem(
                num_vars as Index,
                x_l.as_mut_ptr(),
                x_u.as_mut_ptr(),
                0, // no constraints
                ::std::ptr::null_mut(),
                ::std::ptr::null_mut(),
                0, // no non-zeros in constraint jacobian
                nlp.num_hessian_non_zeros() as Index,
                nlp.indexing_style() as Index,
                Some(Self::eval_f),
                Some(Self::eval_g_none),
                Some(Self::eval_grad_f),
                Some(Self::eval_jac_g_none),
                Some(Self::eval_h))
        };

        let mut mult_x_l = Vec::with_capacity(num_vars);
        mult_x_l.resize(num_vars, 0.0);
        let mut mult_x_u = Vec::with_capacity(num_vars);
        mult_x_u.resize(num_vars, 0.0);
        let x = nlp.initial_point();

        Ipopt {
            nlp_internal,
            nlp_interface: nlp,
            mult_g: Vec::new(),
            mult_x_l,
            mult_x_u,
            x,
            intermediate_callback: None,
        }
    }

    /**
     * Ipopt C API
     */

    /// Evaluate the hessian matrix.
    unsafe extern "C" fn eval_h(
        n: Index,
        x: *mut Number,
        _new_x: Bool,
        obj_factor: Number,
        _m: Index,
        _lambda: *mut Number,
        _new_lambda: Bool,
        nele_hess: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        user_data: ffi::UserDataPtr) -> Bool
    {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        if values.is_null() {
            /* return the structure. */
            nlp.hessian_indices(
                slice::from_raw_parts_mut(irow, nele_hess as usize),
                slice::from_raw_parts_mut(jcol, nele_hess as usize)) as Bool
        } else {
            /* return the values. */
            let result = nlp.hessian_values(
                slice::from_raw_parts(x, n as usize),
                slice::from_raw_parts_mut(values, nele_hess as usize)) as Bool;
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
    pub fn new(nlp: P) -> Self {
        let (mut x_l, mut x_u) = nlp.bounds();
        let (mut g_l, mut g_u) = nlp.constraint_bounds();

        let num_constraints = nlp.num_constraints();
        let num_vars = nlp.num_variables();
        let num_hess_nnz =  nlp.num_hessian_non_zeros();

        let nlp_internal = unsafe {
            ffi::CreateIpoptProblem(
                num_vars as Index,
                x_l.as_mut_ptr(),
                x_u.as_mut_ptr(),
                num_constraints as Index,
                g_l.as_mut_ptr(),
                g_u.as_mut_ptr(),
                nlp.num_constraint_jac_non_zeros() as Index,
                num_hess_nnz as Index,
                nlp.indexing_style() as Index,
                Some(Self::eval_f),
                Some(Self::eval_g),
                Some(Self::eval_grad_f),
                Some(Self::eval_jac_g),
                Some(Self::eval_full_h))
        };

        let mut mult_g = Vec::with_capacity(num_constraints);
        mult_g.resize(num_constraints, 0.0);
        let mut mult_x_l = Vec::with_capacity(num_vars);
        mult_x_l.resize(num_vars, 0.0);
        let mut mult_x_u = Vec::with_capacity(num_vars);
        mult_x_u.resize(num_vars, 0.0);
        let x = nlp.initial_point();

        Ipopt {
            nlp_internal,
            nlp_interface: nlp,
            mult_g,
            mult_x_l,
            mult_x_u,
            x,
            intermediate_callback: None,
        }
    }

    /// Return the multipliers that enforce constraints.
    pub fn constraint_multipliers(&self) -> &[Number] {
        self.mult_g.as_slice()
    }
    
    /**
     * Ipopt C API
     */

    /// Evaluate the constraint function.
    unsafe extern "C" fn eval_g(
        n: Index,
        x: *mut Number,
        _new_x: Bool,
        m: Index,
        g: *mut Number,
        user_data: ffi::UserDataPtr) -> Bool
    {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        nlp.constraint(slice::from_raw_parts(x, n as usize),
                       slice::from_raw_parts_mut(g, m as usize)) as Bool
    }

    /// Evaluate the constraint jacobian.
    unsafe extern "C" fn eval_jac_g(
        n: Index,
        x: *mut Number,
        _new_x: Bool,
        _m: Index,
        nele_jac: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        user_data: ffi::UserDataPtr) -> Bool
    {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        if values.is_null() {
            /* return the structure of the jacobian */
            nlp.constraint_jac_indices(
                slice::from_raw_parts_mut(irow, nele_jac as usize),
                slice::from_raw_parts_mut(jcol, nele_jac as usize)) as Bool
        } else {
            /* return the values of the jacobian of the constraints */
            nlp.constraint_jac_values(
                slice::from_raw_parts(x, n as usize),
                slice::from_raw_parts_mut(values, nele_jac as usize)) as Bool
        }
    }

    /// Evaluate the hessian matrix. Compared to `eval_h` from `NewtonProblem`,
    /// this version includes the constraint hessian.
    unsafe extern "C" fn eval_full_h(
        n: Index,
        x: *mut Number,
        _new_x: Bool,
        obj_factor: Number,
        m: Index,
        lambda: *mut Number,
        _new_lambda: Bool,
        nele_hess: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        user_data: ffi::UserDataPtr) -> Bool
    {
        let nlp = &mut (*(user_data as *mut Ipopt<P>)).nlp_interface;
        if values.is_null() {
            /* return the structure. */
            nlp.hessian_indices(
                slice::from_raw_parts_mut(irow, nele_hess as usize),
                slice::from_raw_parts_mut(jcol, nele_hess as usize)) as Bool
        } else {
            /* return the values. */
            nlp.hessian_values(
                slice::from_raw_parts(x, n as usize),
                obj_factor,
                slice::from_raw_parts(lambda, m as usize),
                slice::from_raw_parts_mut(values, nele_hess as usize)) as Bool
        }
    }
}

impl<P: BasicProblem> Drop for Ipopt<P> {
    fn drop(&mut self) {
        unsafe { ffi::FreeIpoptProblem(self.nlp_internal); }
    }
}

/// Zero-based indexing (C Style) or one-based indexing (Fortran style).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum IndexingStyle {
    CStyle = 0,
    FortranStyle = 1,
}

/// Program return status.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ReturnStatus {
    SolveSucceeded,
    SolvedToAcceptableLevel,
    InfeasibleProblemDetected,
    SearchDirectionBecomesTooSmall,
    DivergingIterates,
    UserRequestedStop,
    FeasiblePointFound,
    MaximumIterationsExceeded,
    RestorationFailed,
    ErrorInStepComputation,
    MaximumCpuTimeExceeded,
    NotEnoughDegreesOfFreedom,
    InvalidProblemDefinition,
    InvalidOption,
    InvalidNumberDetected,
    UnrecoverableException,
    NonIpoptExceptionThrown,
    InsufficientMemory,
    InternalError,
    UnknownError,
}

#[allow(non_snake_case)]
impl ReturnStatus {
    fn new(status: ffi::ApplicationReturnStatus) -> Self {
        use ReturnStatus as RS;
        match status {
            ffi::ApplicationReturnStatus_Solve_Succeeded              => RS::SolveSucceeded,
            ffi::ApplicationReturnStatus_Solved_To_Acceptable_Level   => RS::SolvedToAcceptableLevel,
            ffi::ApplicationReturnStatus_Infeasible_Problem_Detected  => RS::InfeasibleProblemDetected,
            ffi::ApplicationReturnStatus_Search_Direction_Becomes_Too_Small
                => RS::SearchDirectionBecomesTooSmall,
            ffi::ApplicationReturnStatus_Diverging_Iterates            => RS::DivergingIterates,
            ffi::ApplicationReturnStatus_User_Requested_Stop           => RS::UserRequestedStop,
            ffi::ApplicationReturnStatus_Feasible_Point_Found          => RS::FeasiblePointFound,
            ffi::ApplicationReturnStatus_Maximum_Iterations_Exceeded   => RS::MaximumIterationsExceeded,
            ffi::ApplicationReturnStatus_Restoration_Failed            => RS::RestorationFailed,
            ffi::ApplicationReturnStatus_Error_In_Step_Computation     => RS::ErrorInStepComputation,
            ffi::ApplicationReturnStatus_Maximum_CpuTime_Exceeded      => RS::MaximumCpuTimeExceeded,
            ffi::ApplicationReturnStatus_Not_Enough_Degrees_Of_Freedom => RS::NotEnoughDegreesOfFreedom,
            ffi::ApplicationReturnStatus_Invalid_Problem_Definition    => RS::InvalidProblemDefinition,
            ffi::ApplicationReturnStatus_Invalid_Option                => RS::InvalidOption,
            ffi::ApplicationReturnStatus_Invalid_Number_Detected       => RS::InvalidNumberDetected,
            ffi::ApplicationReturnStatus_Unrecoverable_Exception       => RS::UnrecoverableException,
            ffi::ApplicationReturnStatus_NonIpopt_Exception_Thrown     => RS::NonIpoptExceptionThrown,
            ffi::ApplicationReturnStatus_Insufficient_Memory           => RS::InsufficientMemory,
            ffi::ApplicationReturnStatus_Internal_Error                => RS::InternalError,
            _ => RS::UnknownError,
        }
    }
}
