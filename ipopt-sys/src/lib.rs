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
include!(concat!(env!("OUT_DIR"), "/IpStdCInterface.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::slice;
    use std::ffi::CString;

    /// A small structure used to store state between calls to Ipopt NLP callbacks.
    #[derive(Debug)]
    struct UserData {
        g_offset: [Number; 2],
    }
	
    /// Test Ipopt raw bindings. This will also serve as an example of the raw C API.
    #[test]
    fn hs071_test() {
		// rough comparator
		let approx_eq = |a: f64, b: f64| assert!((a-b).abs() < 1e-5, format!("{} vs. {}", a, b));

		/* set the number of variables and allocate space for the bounds */
		let n = 4;  // number of variabes
		let mut x_L = Vec::with_capacity(n); // lower bounds on x
		let mut x_U = Vec::with_capacity(n); // upper bounds on x
		/* set the values for the variable bounds */
        x_L.resize(n, 1.0);
        x_U.resize(n, 5.0);

		/* set the number of constraints and allocate space for the bounds */
		let m = 2usize; // number of constraints
		/* set the values of the constraint bounds */
		let mut g_L = vec![25.0, 40.0]; // lower bounds on g
		let mut g_U = vec![2.0e19, 40.0]; // upper bounds on g

		/* initialize values for the initial point */
		let mut x = vec![1.0, 5.0, 5.0, 1.0];

		/* allocate space to store the bound multipliers at the solution */
        let mut mult_g = Vec::with_capacity(m);
        mult_g.resize(m, 0.0);
		let mut mult_x_L = Vec::with_capacity(n);
        mult_x_L.resize(n, 0.0);
		let mut mult_x_U = Vec::with_capacity(n);
        mult_x_U.resize(n, 0.0);

        let nlp = unsafe {
            /* create the IpoptProblem */
            CreateIpoptProblem(
                n as Index,
			   	x_L.as_mut_ptr(),
			   	x_U.as_mut_ptr(),
				x.as_mut_ptr(),         
				mult_x_L.as_mut_ptr(),  // Initial values for the multipliers for
									    // the lower variable bounds. (if warm start)
				mult_x_U.as_mut_ptr(),  // Initial values for the multipliers for
										// the upper variable bounds. (if warm start)
                m as Index,
			   	g_L.as_mut_ptr(),
			   	g_U.as_mut_ptr(),
				mult_g.as_mut_ptr(),    // Initial values for constraint multipliers.
			   	8, 10, 0, 
                Some(eval_f),
                Some(eval_g),
                Some(eval_grad_f),
                Some(eval_jac_g),
                Some(eval_h))
        };

		/* set some options */
        /* Unfortunately Ipopt strings are non-const, making this awkward. */
        let mut tol_str = CString::new("tol").unwrap();
        let mut mu_strategy_str = CString::new("mu_strategy").unwrap();
        let mut adaptive_str = CString::new("adaptive").unwrap();
        let mut print_lvl_str = CString::new("print_level").unwrap();
        let mut sb_str = CString::new("sb").unwrap();
        let mut yes_str = CString::new("yes").unwrap();

        unsafe {
            AddIpoptIntOption(nlp, (&mut print_lvl_str).as_ptr() as *mut i8, 0);
            AddIpoptNumOption(nlp, (&mut tol_str).as_ptr() as *mut i8, 1e-7);
            AddIpoptStrOption(nlp, (&mut mu_strategy_str).as_ptr() as *mut i8,
                                   (&mut adaptive_str).as_ptr() as *mut i8);
            AddIpoptStrOption(nlp, (&mut sb_str).as_ptr() as *mut i8,
                                   (&mut yes_str).as_ptr() as *mut i8);
            SetIntermediateCallback(nlp, Some(intermediate_cb));
        }

        let mut user_data = UserData { g_offset: [0.0, 0.0] };
        let udata_ptr = (&mut user_data) as *mut UserData;

		/* solve the problem */
		let sol = unsafe {
            IpoptSolve(nlp,
					   udata_ptr as UserDataPtr, // Pointer to user data. This will be passed unmodified
					   							 // to the callback functions.
					   ) // Problem that is to be optimized.
        };

        assert_eq!(sol.status, ApplicationReturnStatus_User_Requested_Stop);

		let mut g = Vec::new();
		g.resize(m, 0.0);

		// Write solutions back to our managed Vecs
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
			AddIpoptStrOption(nlp,
							  (&mut warm_start_str).as_ptr() as *mut i8,
							  (&mut yes_str).as_ptr() as *mut i8);
			AddIpoptNumOption(nlp, (&mut bound_push_str).as_ptr() as *mut i8, 1e-5);
			AddIpoptNumOption(nlp, (&mut bound_frac_str).as_ptr() as *mut i8, 1e-5);
			SetIntermediateCallback(nlp, None);
		}

		let sol = unsafe { IpoptSolve( nlp, udata_ptr as UserDataPtr ) };

		// Write solutions back to our managed Vecs
		x.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.x, n) });
		g.copy_from_slice(unsafe { slice::from_raw_parts(sol.g, m) });
		mult_g.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.mult_g, m) });
		mult_x_L.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.mult_x_L, n) });
		mult_x_U.copy_from_slice(unsafe { slice::from_raw_parts(sol.data.mult_x_U, n) });

		assert_eq!(sol.status, ApplicationReturnStatus_Solve_Succeeded);

		approx_eq(x[0], 1.000000e+00);
		approx_eq(x[1], 4.749269e+00);
		approx_eq(x[2], 3.817510e+00);
		approx_eq(x[3], 1.367870e+00);

		approx_eq(mult_g[0], -5.517016e-01);
		approx_eq(mult_g[1], 1.592915e-01);

		approx_eq(mult_x_L[0], 1.090362e+00);
		approx_eq(mult_x_L[1], 2.664877e-12);
		approx_eq(mult_x_L[2], 3.556758e-12);
		approx_eq(mult_x_L[3], 2.693832e-11);
		approx_eq(mult_x_U[0], 2.498100e-12);
		approx_eq(mult_x_U[1], 4.074104e-11);
		approx_eq(mult_x_U[2], 8.423997e-12);
		approx_eq(mult_x_U[3], 2.755724e-12);

		approx_eq(sol.obj_val, 1.690362e+01);

		/* free allocated memory */
		unsafe { FreeIpoptProblem(nlp); }
    }

    /* Function Implementations */
    unsafe extern "C" fn eval_f(
        n: Index,
        x: *mut Number,
        _new_x: Bool,
        obj_value: *mut Number,
        _user_data: UserDataPtr) -> Bool {
      assert!(n == 4);

      *obj_value = *x.offset(0) * *x.offset(3)
          * (*x.offset(0) + *x.offset(1) + *x.offset(2)) + *x.offset(2);

      true as Bool
    }

    unsafe extern "C" fn eval_grad_f(
        n: Index,
        x: *mut Number,
        _new_x: Bool,
        grad_f: *mut Number,
        _user_data: UserDataPtr) -> Bool {
      assert!(n == 4);

      *grad_f.offset(0) = *x.offset(0) * *x.offset(3)
          + *x.offset(3) * (*x.offset(0) + *x.offset(1) + *x.offset(2));
      *grad_f.offset(1) = *x.offset(0) * *x.offset(3);
      *grad_f.offset(2) = *x.offset(0) * *x.offset(3) + 1.0;
      *grad_f.offset(3) = *x.offset(0) * (*x.offset(0) + *x.offset(1) + *x.offset(2));

      true as Bool
    }

    unsafe extern "C" fn eval_g(
        n: Index,
        x: *mut Number,
        _new_x: Bool,
        m: Index,
        g: *mut Number,
        user_data_ptr: UserDataPtr) -> Bool {
      //struct MyUserData* my_data = user_data;

      assert!(n == 4);
      assert!(m == 2);

      let user_data = &*(user_data_ptr as *mut UserData);
      *g.offset(0) = *x.offset(0) * *x.offset(1) * *x.offset(2) * *x.offset(3) + user_data.g_offset[0];
      *g.offset(1) = *x.offset(0) * *x.offset(0)
          + *x.offset(1) * *x.offset(1)
          + *x.offset(2) * *x.offset(2)
          + *x.offset(3) * *x.offset(3) + user_data.g_offset[1];

      true as Bool
    }

    unsafe extern "C" fn eval_jac_g(
        _n: Index,
        x: *mut Number,
        _new_x: Bool,
        _m: Index,
        _nele_jac: Index,
        iRow: *mut Index,
        jCol: *mut Index,
        values: *mut Number,
        _user_data: UserDataPtr) -> Bool {
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
      }
      else {
        /* return the values of the jacobian of the constraints */

        *values.offset(0) = *x.offset(1)**x.offset(2)**x.offset(3); /* 0,0 */
        *values.offset(1) = *x.offset(0)**x.offset(2)**x.offset(3); /* 0,1 */
        *values.offset(2) = *x.offset(0)**x.offset(1)**x.offset(3); /* 0,2 */
        *values.offset(3) = *x.offset(0)**x.offset(1)**x.offset(2); /* 0,3 */

        *values.offset(4) = 2.0 * *x.offset(0);         /* 1,0 */
        *values.offset(5) = 2.0 * *x.offset(1);         /* 1,1 */
        *values.offset(6) = 2.0 * *x.offset(2);         /* 1,2 */
        *values.offset(7) = 2.0 * *x.offset(3);         /* 1,3 */
      }

     true as Bool 
    }

    unsafe extern "C" fn eval_h(
        _n: Index,
        x: *mut Number,
        _new_x: Bool,
        obj_factor: Number,
        _m: Index,
        lambda: *mut Number,
        _new_lambda: Bool,
        nele_hess: Index,
        iRow: *mut Index,
        jCol: *mut Index,
        values: *mut Number,
        _user_data: UserDataPtr) -> Bool {

      if values.is_null() {
        /* return the structure. This is a symmetric matrix, fill the lower left
         * triangle only. */

        /* the hessian for this problem is actually dense */
        let mut idx = 0;  /* nonzero element counter */
        for row in 0..4 {
          for col in 0..row+1 {
            *iRow.offset(idx) = row;
            *jCol.offset(idx) = col;
            idx += 1;
          }
        }

        assert!(idx == nele_hess as isize);
      }
      else {
        /* return the values. This is a symmetric matrix, fill the lower left
         * triangle only */

        /* fill the objective portion */
        *values.offset(0) = obj_factor * (2.0 * *x.offset(3));               /* 0,0 */

        *values.offset(1) = obj_factor * (*x.offset(3));                 /* 1,0 */
        *values.offset(2) = 0.0;                                   /* 1,1 */

        *values.offset(3) = obj_factor * (*x.offset(3));                 /* 2,0 */
        *values.offset(4) = 0.0;                                   /* 2,1 */
        *values.offset(5) = 0.0;                                   /* 2,2 */

        *values.offset(6) = obj_factor * (2.0 * *x.offset(0) + *x.offset(1) + *x.offset(2)); /* 3,0 */
        *values.offset(7) = obj_factor * (*x.offset(0));                 /* 3,1 */
        *values.offset(8) = obj_factor * (*x.offset(0));                 /* 3,2 */
        *values.offset(9) = 0.0;                                   /* 3,3 */


        /* add the portion for the first constraint */
        *values.offset(1) += *lambda.offset(0) * (*x.offset(2) * *x.offset(3));          /* 1,0 */

        *values.offset(3) += *lambda.offset(0) * (*x.offset(1) * *x.offset(3));          /* 2,0 */
        *values.offset(4) += *lambda.offset(0) * (*x.offset(0) * *x.offset(3));          /* 2,1 */

        *values.offset(6) += *lambda.offset(0) * (*x.offset(1) * *x.offset(2));          /* 3,0 */
        *values.offset(7) += *lambda.offset(0) * (*x.offset(0) * *x.offset(2));          /* 3,1 */
        *values.offset(8) += *lambda.offset(0) * (*x.offset(0) * *x.offset(1));          /* 3,2 */

        /* add the portion for the second constraint */
        *values.offset(0) += *lambda.offset(1) * 2.0;                      /* 0,0 */

        *values.offset(2) += *lambda.offset(1) * 2.0;                      /* 1,1 */

        *values.offset(5) += *lambda.offset(1) * 2.0;                      /* 2,2 */

        *values.offset(9) += *lambda.offset(1) * 2.0;                      /* 3,3 */
      }

      true as Bool
    }

    extern "C" fn intermediate_cb(
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
        _ls_trials: Index,
        _user_data: UserDataPtr) -> Bool {
      if inf_pr < 1e-4 {
          false as Bool
      } else {
          true as Bool
      }
    }
}
