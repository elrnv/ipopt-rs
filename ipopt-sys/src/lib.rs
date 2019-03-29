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

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/ipopt_cnlp.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use std::ffi::CString;
    use std::slice;

    /// A small structure used to store state between calls to Ipopt NLP callbacks.
    #[derive(Debug)]
    struct UserData {
        g_offset: [CNLP_Number; 2],
    }

    /// Test Ipopt raw bindings. This will also serve as an example of the raw C API.
    #[test]
    fn hs071_test() {
        // rough comparator
        let approx_eq =
            |a: f64, b: f64| assert_relative_eq!(a, b, max_relative = 1e-6, epsilon = 1e-14);

        let mut nlp: CNLP_ProblemPtr = ::std::ptr::null_mut();
        let create_status = unsafe {
            /* create the CNLP_Problem */
            cnlp_create_problem(
                &mut nlp as *mut CNLP_ProblemPtr,
                0,
                Some(sizes),
                Some(init),
                Some(bounds),
                Some(eval_f),
                Some(eval_g),
                Some(eval_grad_f),
                Some(eval_jac_g),
                Some(eval_h),
                None,
            )
        };

        assert_eq!(create_status, CNLP_CreateProblemStatus_CNLP_SUCCESS);

        /* Set some options */
        let tol_str = CString::new("tol").unwrap();
        let mu_strategy_str = CString::new("mu_strategy").unwrap();
        let adaptive_str = CString::new("adaptive").unwrap();
        let print_lvl_str = CString::new("print_level").unwrap();
        let sb_str = CString::new("sb").unwrap();
        let yes_str = CString::new("yes").unwrap();

        unsafe {
            cnlp_add_int_option(nlp, print_lvl_str.as_ptr(), 0);
            cnlp_add_num_option(nlp, tol_str.as_ptr(), 1e-7);
            cnlp_add_str_option(nlp, mu_strategy_str.as_ptr(), adaptive_str.as_ptr());
            cnlp_add_str_option(nlp, sb_str.as_ptr(), yes_str.as_ptr());
            cnlp_set_intermediate_callback(nlp, Some(intermediate_cb));
        }

        let mut user_data = UserData {
            g_offset: [0.0, 0.0],
        };
        let udata_ptr = (&mut user_data) as *mut UserData;

        /* solve the problem */
        let sol = unsafe {
            cnlp_solve(
                nlp,
                udata_ptr as CNLP_UserDataPtr, // Pointer to user data. This will be passed unmodified
                                               // to the callback functions.
            ) // Problem that is to be optimized.
        };

        assert_eq!(
            sol.status,
            CNLP_ApplicationReturnStatus_CNLP_USER_REQUESTED_STOP
        );

        let m = 2;
        let n = 4;

        let mut g = vec![0.0; m];

        // Get solution data
        let mut x = vec![0.0; n];
        let mut mult_g = vec![0.0; m];
        let mut mult_x_L = vec![0.0; n];
        let mut mult_x_U = vec![0.0; n];

        x.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.x, n) });
        g.copy_from_slice(unsafe { slice::from_raw_parts(sol.g, m) });
        mult_g.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.mult_g, m) });
        mult_x_L.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.mult_x_L, n) });
        mult_x_U.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.mult_x_U, n) });

        approx_eq(x[0], 1.000000e+00);
        approx_eq(x[1], 4.743000e+00);
        approx_eq(x[2], 3.821150e+00);
        approx_eq(x[3], 1.379408e+00);

        approx_eq(mult_g[0], -5.522936e-01);
        approx_eq(mult_g[1], 1.614685e-01);

        approx_eq(mult_x_L[0], 1.087872e+00);
        approx_eq(mult_x_L[1], 4.635819e-09);
        approx_eq(mult_x_L[2], 9.087447e-09);
        approx_eq(mult_x_L[3], 8.555955e-09);
        approx_eq(mult_x_U[0], 4.470027e-09);
        approx_eq(mult_x_U[1], 4.075231e-07);
        approx_eq(mult_x_U[2], 1.189791e-08);
        approx_eq(mult_x_U[3], 6.398749e-09);

        approx_eq(sol.obj_val, 1.701402e+01);

        // Now we are going to solve this problem again, but with slightly modified
        // constraints.  We change the constraint offset of the first constraint a bit,
        // and resolve the problem using the warm start option.

        user_data.g_offset[0] = 0.2;

        let mut warm_start_str = CString::new("warm_start_init_point").unwrap();
        let mut yes_str = CString::new("yes").unwrap();
        let mut bound_push_str = CString::new("bound_push").unwrap();
        let mut bound_frac_str = CString::new("bound_frac").unwrap();

        unsafe {
            cnlp_add_str_option(
                nlp,
                (&mut warm_start_str).as_ptr() as *mut i8,
                (&mut yes_str).as_ptr() as *mut i8,
            );
            cnlp_add_num_option(nlp, (&mut bound_push_str).as_ptr() as *mut i8, 1e-5);
            cnlp_add_num_option(nlp, (&mut bound_frac_str).as_ptr() as *mut i8, 1e-5);
            cnlp_set_intermediate_callback(nlp, None);
        }

        let sol = unsafe { cnlp_solve(nlp, udata_ptr as CNLP_UserDataPtr) };

        // Write solutions back to our managed Vecs
        x.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.x, n) });
        g.copy_from_slice(unsafe { slice::from_raw_parts(sol.g, m) });
        mult_g.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.mult_g, m) });
        mult_x_L.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.mult_x_L, n) });
        mult_x_U.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.mult_x_U, n) });

        assert_eq!(
            sol.status,
            CNLP_ApplicationReturnStatus_CNLP_SOLVE_SUCCEEDED
        );

        approx_eq(x[0], 1.000000e+00);
        approx_eq(x[1], 4.749269e+00);
        approx_eq(x[2], 3.817510e+00);
        approx_eq(x[3], 1.367870e+00);

        approx_eq(mult_g[0], -5.517016e-01);
        approx_eq(mult_g[1], 1.592915e-01);

        approx_eq(mult_x_L[0], 1.090362e+00);
        approx_eq(mult_x_L[1], 2.664877e-12);
        approx_eq(mult_x_L[2], 3.556758e-12);
        approx_eq(mult_x_L[3], 2.718148e-11);
        approx_eq(mult_x_U[0], 2.498100e-12);
        approx_eq(mult_x_U[1], 4.013263e-11);
        approx_eq(mult_x_U[2], 8.455630e-12);
        approx_eq(mult_x_U[3], 2.755724e-12);

        approx_eq(sol.obj_val, 1.690362e+01);

        /* free allocated memory */
        unsafe {
            cnlp_free_problem(nlp);
        }
    }

    /* Function Implementations */
    unsafe extern "C" fn sizes(
        n: *mut CNLP_Index,
        m: *mut CNLP_Index,
        nnz_jac_g: *mut CNLP_Index,
        nnz_h_lag: *mut CNLP_Index,
        _user_data: CNLP_UserDataPtr,
    ) -> CNLP_Bool {
        *n = 4; // CNLP_Number of variables
        *m = 2; // CNLP_Number of constraints
        *nnz_jac_g = 8; // CNLP_Number of jacobian non-zeros
        *nnz_h_lag = 10; // CNLP_Number of hessian non-zeros
        true as CNLP_Bool
    }

    unsafe extern "C" fn init(
        n: CNLP_Index,
        init_x: CNLP_Bool,
        x: *mut CNLP_Number,
        init_z: CNLP_Bool,
        z_L: *mut CNLP_Number,
        z_U: *mut CNLP_Number,
        m: CNLP_Index,
        init_lambda: CNLP_Bool,
        lambda: *mut CNLP_Number,
        _user_data: CNLP_UserDataPtr,
    ) -> CNLP_Bool {
        assert_eq!(n, 4);
        assert_eq!(m, 2);
        if init_x != 0 {
            /* initialize values for the initial point */
            *x.offset(0) = 1.0;
            *x.offset(1) = 5.0;
            *x.offset(2) = 5.0;
            *x.offset(3) = 1.0;
        }
        if init_z != 0 {
            /* initialize multipliers for the variable bounds */
            for i in 0..4 {
                *z_L.offset(i) = 0.0;
                *z_U.offset(i) = 0.0;
            }
        }
        if init_lambda != 0 {
            *lambda.offset(0) = 0.0;
            *lambda.offset(1) = 0.0;
        }
        true as CNLP_Bool
    }

    unsafe extern "C" fn bounds(
        n: CNLP_Index,
        x_l: *mut CNLP_Number,
        x_u: *mut CNLP_Number,
        m: CNLP_Index,
        g_l: *mut CNLP_Number,
        g_u: *mut CNLP_Number,
        _user_data: CNLP_UserDataPtr,
    ) -> CNLP_Bool {
        assert!(n == 4);
        assert!(m == 2);

        /* Set the values of the constraint bounds */
        *g_l.offset(0) = 25.0;
        *g_l.offset(1) = 40.0;
        *g_u.offset(0) = 2.0e19;
        *g_u.offset(1) = 40.0;

        /* Set the values for the variable bounds */
        for i in 0..n as isize {
            *x_l.offset(i) = 1.0;
            *x_u.offset(i) = 5.0;
        }

        true as CNLP_Bool
    }

    unsafe extern "C" fn eval_f(
        n: CNLP_Index,
        x: *const CNLP_Number,
        _new_x: CNLP_Bool,
        obj_value: *mut CNLP_Number,
        _user_data: CNLP_UserDataPtr,
    ) -> CNLP_Bool {
        assert!(n == 4);

        *obj_value = *x.offset(0) * *x.offset(3) * (*x.offset(0) + *x.offset(1) + *x.offset(2))
            + *x.offset(2);

        true as CNLP_Bool
    }

    unsafe extern "C" fn eval_grad_f(
        n: CNLP_Index,
        x: *const CNLP_Number,
        _new_x: CNLP_Bool,
        grad_f: *mut CNLP_Number,
        _user_data: CNLP_UserDataPtr,
    ) -> CNLP_Bool {
        assert!(n == 4);

        *grad_f.offset(0) = *x.offset(0) * *x.offset(3)
            + *x.offset(3) * (*x.offset(0) + *x.offset(1) + *x.offset(2));
        *grad_f.offset(1) = *x.offset(0) * *x.offset(3);
        *grad_f.offset(2) = *x.offset(0) * *x.offset(3) + 1.0;
        *grad_f.offset(3) = *x.offset(0) * (*x.offset(0) + *x.offset(1) + *x.offset(2));

        true as CNLP_Bool
    }

    unsafe extern "C" fn eval_g(
        n: CNLP_Index,
        x: *const CNLP_Number,
        _new_x: CNLP_Bool,
        m: CNLP_Index,
        g: *mut CNLP_Number,
        user_data_ptr: CNLP_UserDataPtr,
    ) -> CNLP_Bool {
        //struct MyUserData* my_data = user_data;

        assert!(n == 4);
        assert!(m == 2);

        let user_data = &*(user_data_ptr as *mut UserData);
        *g.offset(0) =
            *x.offset(0) * *x.offset(1) * *x.offset(2) * *x.offset(3) + user_data.g_offset[0];
        *g.offset(1) = *x.offset(0) * *x.offset(0)
            + *x.offset(1) * *x.offset(1)
            + *x.offset(2) * *x.offset(2)
            + *x.offset(3) * *x.offset(3)
            + user_data.g_offset[1];

        true as CNLP_Bool
    }

    unsafe extern "C" fn eval_jac_g(
        _n: CNLP_Index,
        x: *const CNLP_Number,
        _new_x: CNLP_Bool,
        _m: CNLP_Index,
        _nele_jac: CNLP_Index,
        iRow: *mut CNLP_Index,
        jCol: *mut CNLP_Index,
        values: *mut CNLP_Number,
        _user_data: CNLP_UserDataPtr,
    ) -> CNLP_Bool {
        if values.is_null() {
            /* return the structure of the jacobian */

            /* this particular jacobian is dense */
            *iRow.offset(0) = 0;
            *jCol.offset(0) = 0;
            *iRow.offset(1) = 0;
            *jCol.offset(1) = 1;
            *iRow.offset(2) = 0;
            *jCol.offset(2) = 2;
            *iRow.offset(3) = 0;
            *jCol.offset(3) = 3;
            *iRow.offset(4) = 1;
            *jCol.offset(4) = 0;
            *iRow.offset(5) = 1;
            *jCol.offset(5) = 1;
            *iRow.offset(6) = 1;
            *jCol.offset(6) = 2;
            *iRow.offset(7) = 1;
            *jCol.offset(7) = 3;
        } else {
            /* return the values of the jacobian of the constraints */

            *values.offset(0) = *x.offset(1) * *x.offset(2) * *x.offset(3); /* 0,0 */
            *values.offset(1) = *x.offset(0) * *x.offset(2) * *x.offset(3); /* 0,1 */
            *values.offset(2) = *x.offset(0) * *x.offset(1) * *x.offset(3); /* 0,2 */
            *values.offset(3) = *x.offset(0) * *x.offset(1) * *x.offset(2); /* 0,3 */

            *values.offset(4) = 2.0 * *x.offset(0); /* 1,0 */
            *values.offset(5) = 2.0 * *x.offset(1); /* 1,1 */
            *values.offset(6) = 2.0 * *x.offset(2); /* 1,2 */
            *values.offset(7) = 2.0 * *x.offset(3); /* 1,3 */
        }

        true as CNLP_Bool
    }

    unsafe extern "C" fn eval_h(
        _n: CNLP_Index,
        x: *const CNLP_Number,
        _new_x: CNLP_Bool,
        obj_factor: CNLP_Number,
        _m: CNLP_Index,
        lambda: *const CNLP_Number,
        _new_lambda: CNLP_Bool,
        nele_hess: CNLP_Index,
        iRow: *mut CNLP_Index,
        jCol: *mut CNLP_Index,
        values: *mut CNLP_Number,
        _user_data: CNLP_UserDataPtr,
    ) -> CNLP_Bool {
        if values.is_null() {
            /* return the structure. This is a symmetric matrix, fill the lower left
             * triangle only. */

            /* the hessian for this problem is actually dense */
            let mut idx = 0; /* nonzero element counter */
            for row in 0..4 {
                for col in 0..row + 1 {
                    *iRow.offset(idx) = row;
                    *jCol.offset(idx) = col;
                    idx += 1;
                }
            }

            assert!(idx == nele_hess as isize);
        } else {
            /* return the values. This is a symmetric matrix, fill the lower left
             * triangle only */

            /* fill the objective portion */
            *values.offset(0) = obj_factor * (2.0 * *x.offset(3)); /* 0,0 */

            *values.offset(1) = obj_factor * (*x.offset(3)); /* 1,0 */
            *values.offset(2) = 0.0; /* 1,1 */

            *values.offset(3) = obj_factor * (*x.offset(3)); /* 2,0 */
            *values.offset(4) = 0.0; /* 2,1 */
            *values.offset(5) = 0.0; /* 2,2 */

            *values.offset(6) = obj_factor * (2.0 * *x.offset(0) + *x.offset(1) + *x.offset(2)); /* 3,0 */
            *values.offset(7) = obj_factor * (*x.offset(0)); /* 3,1 */
            *values.offset(8) = obj_factor * (*x.offset(0)); /* 3,2 */
            *values.offset(9) = 0.0; /* 3,3 */

            /* add the portion for the first constraint */
            *values.offset(1) += *lambda.offset(0) * (*x.offset(2) * *x.offset(3)); /* 1,0 */

            *values.offset(3) += *lambda.offset(0) * (*x.offset(1) * *x.offset(3)); /* 2,0 */
            *values.offset(4) += *lambda.offset(0) * (*x.offset(0) * *x.offset(3)); /* 2,1 */

            *values.offset(6) += *lambda.offset(0) * (*x.offset(1) * *x.offset(2)); /* 3,0 */
            *values.offset(7) += *lambda.offset(0) * (*x.offset(0) * *x.offset(2)); /* 3,1 */
            *values.offset(8) += *lambda.offset(0) * (*x.offset(0) * *x.offset(1)); /* 3,2 */

            /* add the portion for the second constraint */
            *values.offset(0) += *lambda.offset(1) * 2.0; /* 0,0 */

            *values.offset(2) += *lambda.offset(1) * 2.0; /* 1,1 */

            *values.offset(5) += *lambda.offset(1) * 2.0; /* 2,2 */

            *values.offset(9) += *lambda.offset(1) * 2.0; /* 3,3 */
        }

        true as CNLP_Bool
    }

    extern "C" fn intermediate_cb(
        _alg_mod: CNLP_AlgorithmMode,
        _iter_count: CNLP_Index,
        _obj_value: CNLP_Number,
        inf_pr: CNLP_Number,
        _inf_du: CNLP_Number,
        _mu: CNLP_Number,
        _d_norm: CNLP_Number,
        _regularization_size: CNLP_Number,
        _alpha_du: CNLP_Number,
        _alpha_pr: CNLP_Number,
        _ls_trials: CNLP_Index,
        _user_data: CNLP_UserDataPtr,
    ) -> CNLP_Bool {
        if inf_pr < 1e-4 {
            false as CNLP_Bool
        } else {
            true as CNLP_Bool
        }
    }
}
