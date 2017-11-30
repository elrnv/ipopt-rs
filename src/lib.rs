extern crate ipopt_sys as ffi;

use ffi::{Index, Number, Bool, Int};
use std::ffi::CString;
use std::slice;

/// The non-linear problem to be solved by Ipopt. This trait specifies all the
/// information needed to construct the unconstrained optimization problem (although the
/// variables are allowed to be bounded).
/// In the callbacks within, `x` is the independent variable and must be the same size
/// as returned by `num_variables`.
/// Each of the callbacks required during interior point iterations are allowed to fail.
/// In case of failure to produce values, simply return `false` where applicable.
/// This feature could be used to tell Ipopt to try smaller perturbations for `x` for
/// instance.
pub trait UnconstrainedProblem {
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

/// An extension to the `UnconstrainedProblem` trait that enables full Newton iterations in
/// Ipopt. If this trait is NOT implemented by your problem, Ipopt will be set to perform
/// [Quasi-Newton Approximation](https://www.coin-or.org/Ipopt/documentation/node31.html) for
/// second derivatives.
/// This interface asks for the Hessian matrix in sparse triplet form.
pub trait NewtonProblem : UnconstrainedProblem {
    /// Number of non-zeros in the Hessian matrix. This includes the constraint hessian if one is
    /// provided.
    fn num_hessian_non_zeros(&self) -> usize;
    /// Hessian indices. These are the row and column indices of the non-zeros
    /// in the sparse representation of the matrix.
    /// This is a symmetric matrix, fill the lower left triangular half only.
    /// If your problem is constrained (i.e. you are ultimately implementing `Problem`),
    /// ensure that you provide coordinates for non-zeros of the constraint hessian as well.
    /// This function is internally called by Ipopt callback `eval_h`.
    fn hessian_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Objective Hessian values. Each value must correspond to the `row` and `column` as
    /// specified in `hessian_indices`.
    /// This is a symmetric matrix, fill the lower left triangular half only.
    /// This function is internally called by Ipopt callback `eval_h` and each value is
    /// premultiplied by `Ipopt`'s `obj_factor` as necessary.
    fn objective_hessian_values(&mut self, x: &[Number], vals: &mut [Number]) -> bool;
}

/// Extends the `NewtonProblem` trait to enable equality and inequality constraints.
/// Equality constraints are enforce by setting the lower and upper bounds for the constraint to
/// the same value.
/// This type of problem is the target use case for Ipopt.
/// NOTE: Although its possible to run Quasi-Newton iterations on a constrained problem, it
/// doesn't perform well in general, which is the reason this trait inherits `NewtonProblem`
/// instead of `UnconstrainedProblem`. However, you may still enable L-BFGS explicitly by
/// setting the "hessian_approximation" Ipopt option to "limited-memory", in which case you should
/// simply return `false` in the implementation for `NewtonProblem` callbacks.
pub trait Problem : NewtonProblem {
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
    // TODO: veriy whether this function is called only once when the `Ipopt` object is created.
    /// This function is internally called by Ipopt callback `eval_jac_g`.
    fn constraint_jac_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Constraint jacobian values. Each value must correspond to the `row` and
    /// `column` as specified in `constraint_jac_indices`.
    /// This function is internally called by Ipopt callback `eval_jac_g`.
    fn constraint_jac_values(&mut self, x: &[Number], vals: &mut [Number]) -> bool;
    /// Constraint Hessian values. Each value must correspond to the `row` and `column` as
    /// specified in `hessian_indices`.
    /// This is a symmetric matrix, fill the lower left triangular half only.
    /// Compared to `objective_hessian_values` from `NewtonProblem`, this version requires
    /// you to specify the constraint hessian. You should multiply the constraint hessian
    /// values by the corresponding values in `lambda` (the Lagrange multiplier).
    /// This function is internally called by Ipopt callback `eval_h`.
    fn constraint_hessian_values(&mut self,
                                 x: &[Number],
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

impl<'a> From<f64> for IpoptOption<'a> {
    fn from(opt: f64) -> Self {
        IpoptOption::Num(opt)
    }
}

impl<'a> From<&'a str> for IpoptOption<'a> {
    fn from(opt: &'a str) -> Self {
        IpoptOption::Str(opt)
    }
}

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

pub struct Ipopt<P: UnconstrainedProblem> {
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
    /// Temporary constraint hessian values. Used to decouple computation of constraint
    /// hessian from the objective hessian as required by Ipopt.
    temp_constraint_hess_vals: Vec<Number>
}


impl<P: UnconstrainedProblem> Ipopt<P> {
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
                None,
                Some(Self::eval_grad_f),
                None,
                None)
        };

        let mut mult_x_l = Vec::with_capacity(num_vars);
        mult_x_l.resize(num_vars, 0.0);
        let mut mult_x_u = Vec::with_capacity(num_vars);
        mult_x_u.resize(num_vars, 0.0);
        let x = nlp.initial_point();

        assert!(Self::set_ipopt_option(nlp_internal,
                                       "hessian_approximation",
                                       "limited_memory"));
        Ipopt {
            nlp_internal,
            nlp_interface: nlp,
            mult_g: Vec::new(),
            mult_x_l,
            mult_x_u,
            x,
            intermediate_callback: None,
            temp_constraint_hess_vals: Vec::new(),
        }
    }

    /// Get an immutable reference to the provided NLP object.
    pub fn nlp(&self) -> &P {
        &self.nlp_interface
    }
    
    /// Get a mutable reference to the provided NLP object.
    pub fn nlp_mut(&mut self) -> &mut P {
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
    pub fn set_intermediate_callback<F>(&mut self, cb: IntermediateCallback<P>)
        where P: UnconstrainedProblem,
    {
        self.intermediate_callback = Some(cb);
        unsafe {
            ffi::SetIntermediateCallback(self.nlp_internal, Some(Self::intermediate_cb));
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
    pub fn bound_multipliers(&self) -> (&Vec<Number>, &Vec<Number>) {
        (&self.mult_x_l, &self.mult_x_u)
    }

    /// Return the multipliers that enforce the variable bounds.
    pub fn solution(&self) -> &Vec<Number> {
        &self.x
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
                regularization_size, alpha_du, alpha_pr, ls_trials);
        }
        true as Bool
    }
}

impl<P: NewtonProblem> Ipopt<P> {
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
                None,
                Some(Self::eval_grad_f),
                None,
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
            temp_constraint_hess_vals: Vec::new(),
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
            let result = nlp.objective_hessian_values(
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


impl<P: Problem> Ipopt<P> {
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

        let mut temp_constraint_hess_vals = Vec::with_capacity(num_hess_nnz);
        temp_constraint_hess_vals.resize(num_hess_nnz, 0.0);

        Ipopt {
            nlp_internal,
            nlp_interface: nlp,
            mult_g,
            mult_x_l,
            mult_x_u,
            x,
            intermediate_callback: None,
            temp_constraint_hess_vals,
        }
    }

    /// Return the multipliers that enforce constraints.
    pub fn constraint_multipliers(&self) -> &Vec<Number> {
        &self.mult_g
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

    /// Helper utility to set a Vec<Number> to be all zeros.
    fn zero_vec(buf: &mut Vec<Number>) {
        // We could use write_bytes or memset here but this method is stable and has no significant
        // performance cost.
        let n = buf.len();
        buf.clear();
        buf.resize(n, 0.0);
    }

    /// Evaluate the hessian matrix. Compared to `eval_h`, this version includes the constraint
    /// hessian.
    unsafe extern "C" fn eval_full_h(
        n: Index,
        x: *mut Number,
        new_x: Bool,
        obj_factor: Number,
        m: Index,
        lambda: *mut Number,
        new_lambda: Bool,
        nele_hess: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        user_data: ffi::UserDataPtr) -> Bool
    {
        let obj_result = Self::eval_h(n,x,new_x,obj_factor,
                                      m,lambda,new_lambda,
                                      nele_hess,irow,jcol,values,user_data);

        let ip = &mut (*(user_data as *mut Ipopt<P>));
        let constraint_result = if !values.is_null() {
            Self::zero_vec(&mut ip.temp_constraint_hess_vals);
            let constraint_result =
                ip.nlp_interface.constraint_hessian_values(
                    slice::from_raw_parts(x, n as usize),
                    slice::from_raw_parts(lambda, m as usize),
                    slice::from_raw_parts_mut(ip.temp_constraint_hess_vals.as_mut_ptr(),
                                              nele_hess as usize)) as Bool;

            // add constraint hessian
            let start_idx = ip.nlp_interface.indexing_style() as usize;
            for i in start_idx..nele_hess as usize {
                *values.offset(i as isize) += ip.temp_constraint_hess_vals[i];
            }
            constraint_result
        } else { 1 };

        constraint_result * obj_result
    }
}

impl<P: UnconstrainedProblem> Drop for Ipopt<P> {
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

#[cfg(test)]
mod tests {
    use super::*;

    struct NLP {
        g_offset: [f64; 2],
    }
    impl UnconstrainedProblem for NLP {
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
    impl NewtonProblem for NLP {
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
		fn objective_hessian_values(&mut self,
                                    x: &[Number],
                                    vals: &mut [Number]) -> bool {

			vals[0] = 2.0*x[3];                 /* 0,0 */

			vals[1] = x[3];                     /* 1,0 */
			vals[2] = 0.0;                      /* 1,1 */

			vals[3] = x[3];                     /* 2,0 */
			vals[4] = 0.0;                      /* 2,1 */
			vals[5] = 0.0;                      /* 2,2 */

			vals[6] = 2.0*x[0] + x[1] + x[2];   /* 3,0 */
			vals[7] = x[0];                     /* 3,1 */
			vals[8] = x[0];                     /* 3,2 */
			vals[9] = 0.0;                      /* 3,3 */
            true
        }
    }
    impl Problem for NLP {
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
		fn constraint_hessian_values(&mut self,
                                     x: &[Number],
                                     lambda: &[Number],
                                     vals: &mut [Number]) -> bool {
			/* add the portion for the first constraint */
			vals[1] += lambda[0] * (x[2] * x[3]);          /* 1,0 */

			vals[3] += lambda[0] * (x[1] * x[3]);          /* 2,0 */
			vals[4] += lambda[0] * (x[0] * x[3]);          /* 2,1 */

			vals[6] += lambda[0] * (x[1] * x[2]);          /* 3,0 */
			vals[7] += lambda[0] * (x[0] * x[2]);          /* 3,1 */
			vals[8] += lambda[0] * (x[0] * x[1]);          /* 3,2 */

			/* add the portion for the second constraint */
			vals[0] += lambda[1] * 2.0;                      /* 0,0 */

			vals[2] += lambda[1] * 2.0;                      /* 1,1 */

			vals[5] += lambda[1] * 2.0;                      /* 2,2 */

			vals[9] += lambda[1] * 2.0;                      /* 3,3 */
            true
		}
    }

    #[test]
    fn hs071_test() {
        let nlp = NLP { g_offset: [0.0, 0.0] };
        let mut ipopt = Ipopt::new(nlp);
        ipopt.set_option("tol", 1e-7);
        ipopt.set_option("mu_strategy", "adaptive");
        let (_,obj) = ipopt.solve();
		let print_sol = |ipopt: &Ipopt<NLP>, obj| {
			println!("\n\nSolution of the primal variables, x");
			for (i, x_val) in ipopt.solution().iter().enumerate() {
				println!("x[{}] = {:e}", i, x_val);
			}

			println!("\n\nSolution of the constraint multipliers, lambda");
			for (i, mult_g_val) in ipopt.constraint_multipliers().iter().enumerate() {
				println!("lambda[{}] = {:e}", i, mult_g_val);
			}
            let (mult_x_l, mult_x_u) = ipopt.bound_multipliers();
			println!("\n\nSolution of the bound multipliers, z_L and z_U");
			for (i, mult_x_l_val) in mult_x_l.iter().enumerate() {
				println!("z_L[{}] = {:e}", i, mult_x_l_val);
			}
			for (i, mult_x_u_val) in mult_x_u.iter().enumerate() {
				println!("z_U[{}] = {:e}", i, mult_x_u_val);
			}

			println!("\n\nObjective value\nf(x*) = {:e}", obj);
        };

        print_sol(&ipopt, obj);

        ipopt.nlp_mut().g_offset[0] = 0.2;
        ipopt.set_option("warm_start_init_point", "yes");
        ipopt.set_option("bound_push", 1e-5);
        ipopt.set_option("bound_frac", 1e-5);
        let (_, obj) = ipopt.solve();
        print_sol(&ipopt, obj);
    }
}
