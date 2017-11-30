extern crate ipopt;

use ipopt::*;

struct NLP {
    g_offset: [f64; 2],
}

impl BasicProblem for NLP {
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
impl ConstrainedProblem for NLP {
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
    ipopt.set_option("sb", "yes"); // suppress license message
    ipopt.set_option("print_level", 0); // suppress debug output
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
