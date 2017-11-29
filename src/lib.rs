extern crate ipopt_sys as ffi;

use ffi::{Index, Number, Bool, Int};
use std::ffi::CString;
use std::slice;

/// The non-linear problem to be solved by Ipopt. This trait speficifies all the
/// information needed to construct the constrained optimization problem.
/// In the callbacks within, `x` is the independent variable and must be the same size
/// as returned by `num_variables`.
/// Each of the callbacks required during interior point iterations are allowed to fail.
/// In case of failure to produce values, simply return `false` where applicable.
/// This feature could be used to tell Ipopt to try smaller perturbations for `x` for
/// instance.
pub trait Problem {
    /// Specify the indexing style used for arrays in this problem.
    /// (Default is zero-based)
    fn indexing_style(&self) -> IndexingStyle { IndexingStyle::CStyle }
    /// Total number of variables of the non-linear problem.
    fn num_variables(&self) -> usize;
    /// Number of equality and inequality constraints.
    fn num_constraints(&self) -> usize;
    /// Number of non-zeros in the constraint jacobian.
    fn num_constraint_jac_non_zeros(&self) -> usize;
    /// Number of non-zeros in the Hessian matrix.
    fn num_hessian_non_zeros(&self) -> usize;

    /// Specify the pair of variable bounds `(lower, upper)`.
    /// The returned `Vec`s must have the same size as `num_variables`.
    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>);
    /// Specify the pair of bounds `(lower, upper)` on the value of the constraint
    /// function.
    /// The returned `Vec`s must each have the same size as `num_constraints`.
    fn constraint_bounds(&self) -> (Vec<Number>, Vec<Number>);

    /// Construct the initial guess for Ipopt to start with.
    /// The returned `Vec` must have the same size as `num_variables`.
    fn initial_point(&self) -> Vec<Number>;

    /// Objective function. This is the function being minimized.
    /// This function is internally called by Ipopt callback `eval_f`.
    fn objective(&mut self, x: &[Number], obj: &mut Number) -> bool;
    /// Constraint function. This gives the value of each constraint.
    /// The output slice `g` is guaranteed to be the same size as `num_constraints`.
    /// This function is internally called by Ipopt callback `eval_g`.
    fn constraint(&mut self, x: &[Number], g: &mut [Number]) -> bool;
    /// Gradient of the objective function.
    /// This function is internally called by Ipopt callback `eval_grad_f`.
    fn objective_grad(&mut self, x: &[Number], grad_f: &mut [Number]) -> bool;
    /// Constraint jacobian indices. These are the row and column indices of the
    /// non-zeros in the sparse representation of the matrix.
    // TODO: veriy whether this function is called only once when the `Ipopt` object is created.
    /// This function is internally called by Ipopt callback `eval_jac_g`.
    fn constraint_jac_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Constraint jacobian values. Each value must correspond to the `row` and
    /// `column` as specified in `constraint_jac_indices`.
    /// This function is internally called by Ipopt callback `eval_jac_g`.
    fn constraint_jac_values(&mut self, x: &[Number], vals: &mut [Number]) -> bool;
    /// Hessian indices. These are the `(row,column)` pairs of the non-zeros
    /// in the sparse representation of the matrix.
    // TODO: veriy whether this function is called only once when the `Ipopt` object is created.
    /// This is a symmetric matrix, fill the lower left triangular half only.
    /// This function is internally called by Ipopt callback `eval_h`.
    fn hessian_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool;
    /// Hessian values. Each value must correspond to the `row` and `column` as
    /// specified in `hessian_indices`.
    /// This is a symmetric matrix, fill the lower left triangular half only.
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

pub struct Ipopt<P: Problem> {
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

impl<P: Problem> Ipopt<P> {
    pub fn new(nlp: P) -> Self {
        let (mut x_l, mut x_u) = nlp.variable_bounds();
        let (mut g_l, mut g_u) = nlp.constraint_bounds();

        let num_constraints = nlp.num_constraints();
        let num_vars = nlp.num_variables();

        let nlp_internal = unsafe {
            ffi::CreateIpoptProblem(
                num_vars as Index,
                x_l.as_mut_ptr(),
                x_u.as_mut_ptr(),
                num_constraints as Index,
                g_l.as_mut_ptr(),
                g_u.as_mut_ptr(),
                nlp.num_constraint_jac_non_zeros() as Index,
                nlp.num_hessian_non_zeros() as Index,
                nlp.indexing_style() as Index,
                Some(Self::eval_f),
                Some(Self::eval_g),
                Some(Self::eval_grad_f),
                Some(Self::eval_jac_g),
                Some(Self::eval_h))
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

    /// Set an Ipopt option.
    pub fn set_option<'a, O>(&mut self, name: &str, option: O) -> Option<&mut Self>
        where O: Into<IpoptOption<'a>>
    {
        let success = unsafe {
            // Convert the input name string to a `char *` C type
            let mut name_cstr = CString::new(name).unwrap();
            let name_cstr = (&mut name_cstr).as_ptr() as *mut i8; // this is `char *`
            // Match option to one of the three types of options Ipopt can receive.
            match option.into() {
                IpoptOption::Num(opt) =>
                    ffi::AddIpoptNumOption(self.nlp_internal, name_cstr, opt as Number),
                IpoptOption::Str(opt) => {
                    // Convert option string to `char *`
                    let mut opt_cstr = CString::new(opt).unwrap();
                    let opt_cstr = (&mut opt_cstr).as_ptr() as *mut i8;
                    ffi::AddIpoptStrOption(self.nlp_internal, name_cstr, opt_cstr)
                },
                IpoptOption::Int(opt) =>
                    ffi::AddIpoptIntOption(self.nlp_internal, name_cstr, opt as Int),
            }
        } != 0; // converts Ipopt Bool to Rust bool

        if success { Some(self) } else { None }
    }

    /// Set intermediate callback.
    pub fn set_intermediate_callback<F>(&mut self, cb: IntermediateCallback<P>)
        where P: Problem,
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
                self.mult_g.as_mut_ptr(),
                self.mult_x_l.as_mut_ptr(),
                self.mult_x_u.as_mut_ptr(),
                udata_ptr as ffi::UserDataPtr)
        });

        (status, objective_value)
    }

    /// Return the multipliers that enforce constraints.
    pub fn constraint_multipliers(&self) -> &Vec<Number> {
        &self.mult_g
    }

    /// Return the multipliers that enforce the variable bounds.
    pub fn variable_bound_multipliers(&self) -> (&Vec<Number>, &Vec<Number>) {
        (&self.mult_x_l, &self.mult_x_u)
    }

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

    /// Evaluate the hessian matrix.
    unsafe extern "C" fn eval_h(
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
        let ipopt = &mut (*(user_data as *mut Ipopt<P>));
        if let Some(callback) = ipopt.intermediate_callback {
            (callback)(
                &mut ipopt.nlp_interface,
                alg_mod, iter_count, obj_value,
                inf_pr, inf_du, mu, d_norm,
                regularization_size, alpha_du, alpha_pr, ls_trials);
        }
        true as Bool
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
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
