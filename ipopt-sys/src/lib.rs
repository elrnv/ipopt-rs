#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/IpStdCInterface.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use std::ffi::CString;
	
    /// Test Ipopt raw bindings. This will also serve as an example of the raw C API.
    #[test]
    fn hs071_test() {
		/* set the number of variables and allocate space for the bounds */
		let n = 4;  // number of variabes
		let mut x_L = Vec::with_capacity(n); // lower bounds on x
		let mut x_U = Vec::with_capacity(n); // upper bounds on x
		/* set the values for the variable bounds */
        x_L.resize(n, 1.0);
        x_U.resize(n, 5.0);

		/* set the number of constraints and allocate space for the bounds */
		let m = 2; // number of constraints
		/* set the values of the constraint bounds */
		let mut g_L = vec![25.0, 40.0]; // lower bounds on g
		let mut g_U = vec![2.0e19, 40.0]; // upper bounds on g

        let nlp = unsafe {
            /* create the IpoptProblem */
            CreateIpoptProblem(
                n as Index, x_L.as_mut_ptr(), x_U.as_mut_ptr(),
                m, g_L.as_mut_ptr(), g_U.as_mut_ptr(), 8, 10, 0, 
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

        unsafe {
            AddIpoptNumOption(nlp, (&mut tol_str).as_ptr() as *mut i8, 1e-9);
            AddIpoptStrOption(nlp,
                                     (&mut mu_strategy_str).as_ptr() as *mut i8,
                                     (&mut adaptive_str).as_ptr() as *mut i8);
        }

		/* initialize values for the initial point */
		let mut x = vec![1.0, 5.0, 5.0, 1.0];

		/* allocate space to store the bound multipliers at the solution */
		let mut mult_x_L = Vec::with_capacity(n);
        mult_x_L.resize(n, 0.0);
		let mut mult_x_U = Vec::with_capacity(n);
        mult_x_U.resize(n, 0.0);

		let mut obj = 0.0; // objective value as output
		/* solve the problem */
		let status = unsafe {
            IpoptSolve(
                nlp,                     // Problem that is to be optimized.
                x.as_mut_ptr(),          // Input: Starting point; Output: Final solution
                ptr::null_mut(),         // Values of constraint g at final point (output only)
                &mut obj as *mut Number, // Final value of the objective function (output only)
                ptr::null_mut(),         // Input: Initial values for constraint multipliers.
                                         // Output: Final multipliers for constraints.
                mult_x_L.as_mut_ptr(),   // Input: Initial values for the multipliers for
                                         //        the lower variable bounds. (if warm start)
                                         // Output: Final multipliers for lower variable bounds
                mult_x_U.as_mut_ptr(),   // Input: Initial values for the multipliers for
                                         //        the upper variable bounds. (if warm start)
                                         // Output: Final multipliers for upper variable bounds.
                ptr::null_mut())         // Pointer to user data. This will be passed unmodified
                                         // to the callback functions.
        };

		if status == ApplicationReturnStatus_Solve_Succeeded {
			println!("\n\nSolution of the primal variables, x");
			for (i, x_val) in x.iter().enumerate() {
				println!("x[{}] = {}", i, x_val); 
            }

			println!("\n\nSolution of the bound multipliers, z_L and z_U");
			for (i, x_L_val) in mult_x_L.iter().enumerate() {
				println!("z_L[{}] = {}", i, x_L_val); 
            }
			for (i, x_U_val) in mult_x_U.iter().enumerate() {
				println!("z_U[{}] = {}", i, x_U_val); 
            }

            println!("\n\nObjective value\nf(x*) = {}", obj); 
		}

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
        _user_data: UserDataPtr) -> Bool {
      //struct MyUserData* my_data = user_data;

      assert!(n == 4);
      assert!(m == 2);

      *g.offset(0) = *x.offset(0) * *x.offset(1) * *x.offset(2) * *x.offset(3);// + my_data->g_offset.offset(0);
      *g.offset(1) = *x.offset(0)**x.offset(0) + *x.offset(1)**x.offset(1) + *x.offset(2)**x.offset(2) + *x.offset(3)**x.offset(3);// + my_data->g_offse*t.offset(1);

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
        iter_count: Index,
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

      println!("Testing intermediate callback in iteration {}", iter_count);
      if inf_pr < 1e-4 {
          false as Bool
      } else {
          true as Bool
      }
    }
}
