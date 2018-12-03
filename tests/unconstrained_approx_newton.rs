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
extern crate ipopt;
#[macro_use] extern crate approx;

use ipopt::*;

struct NLP {
}

impl BasicProblem for NLP {
    fn num_variables(&self) -> usize { 2 }
    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        x_l.swap_with_slice(vec![-1e20; 2].as_mut_slice());
        x_u.swap_with_slice(vec![1e20; 2].as_mut_slice());
        true
    }
    fn initial_point(&self) -> Vec<Number> { vec![0.0, 0.0] }
    fn objective(&mut self, x: &[Number], obj: &mut Number) -> bool {
        *obj = (x[0] - 1.0)*(x[0] - 1.0) + (x[1] - 1.0)*(x[1] - 1.0);
        true
    }
    fn objective_grad(&mut self, x: &[Number], grad_f: &mut [Number]) -> bool {
        grad_f[0] = 2.0*(x[0] - 1.0);
        grad_f[1] = 2.0*(x[1] - 1.0);
        true
    }
}

#[test]
fn quadratic_test() {
    let nlp = NLP { };
    let mut ipopt = Ipopt::new_unconstrained(nlp).unwrap();
    ipopt.set_option("tol", 1e-9);
    ipopt.set_option("mu_strategy", "adaptive");
    ipopt.set_option("sb", "yes"); // suppress license message
    ipopt.set_option("print_level", 0); // suppress debug output
    {
        let SolveResult {
            solver_data: SolverDataMut {
                primal_variables: x,
                ..
            },
            objective_value: obj,
            status,
            ..
        } = ipopt.solve();
        assert_eq!(status, SolveStatus::SolveSucceeded);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(obj, 0.0, epsilon = 1e-10);
    }

    ipopt.set_option("warm_start_init_point", "yes");
    {
        let SolveResult {
            solver_data: SolverDataMut {
                primal_variables: x,
                ..
            },
            objective_value: obj,
            status,
            ..
        } = ipopt.solve();
        assert_eq!(status, SolveStatus::SolveSucceeded);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(obj, 0.0, epsilon = 1e-10);
    }
}
