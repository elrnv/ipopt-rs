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

/**
 * This test ensures that you can change the problem size between solves.
 * This problem demonstrates the flexibility of the current Rust API. Although the problem size
 * changes, we can still use the solution from the previous solve for the new problem with a larger
 * problem size. This shows that warm starts can be very problem specific.
 */
use approx::assert_relative_eq;
use std::cell::RefCell;

use ipopt::*;

struct NLP {
    z: bool,
    iterations: usize,
    x_start: RefCell<Vec<f64>>, // Save variable results for warm start
}

impl NLP {
    fn count_iterations_cb(&mut self, data: IntermediateCallbackData) -> bool {
        self.iterations = data.iter_count as usize;
        true
    }

    // Save a solution for warm start. Yes this is boilerplate for warm starts. Including this
    // functionality into the crate is future work.
    fn save_solution_for_warm_start(&mut self, solution: Solution) {
        self.x_start.borrow_mut().clear();
        self.x_start
            .borrow_mut()
            .extend_from_slice(&solution.primal_variables);
    }
}

impl BasicProblem for NLP {
    fn num_variables(&self) -> usize {
        let n = if self.z { 3 } else { 2 };

        // Ensure that our warm start vectors have the right size.
        // Note that padding the array with zeros is not always the right thing to do, it is
        // highly problem specific.
        //
        // Interior mutability is required here because the API only gives a const reference.
        // Although this makes it awkward for this use case, it is very prohibitive to require mut
        // references in other cases.

        self.x_start.borrow_mut().resize(n, 0.0);

        n
    }
    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        let n = x_l.len();
        assert_eq!(x_l.len(), x_u.len());
        assert_eq!(n, self.num_variables());
        x_l.swap_with_slice(vec![-1e20; n].as_mut_slice());
        x_u.swap_with_slice(vec![1e20; n].as_mut_slice());
        true
    }
    fn initial_point(&self, x: &mut [Number]) -> bool {
        let n = x.len();
        assert_eq!(n, self.num_variables());
        x.copy_from_slice(&self.x_start.borrow());
        true
    }
    fn objective(&self, x: &[Number], obj: &mut Number) -> bool {
        *obj = 0.0;
        for &val in x.iter() {
            *obj += (val - 1.0) * (val - 1.0);
        }
        *obj *= 0.5;
        true
    }
    fn objective_grad(&self, x: &[Number], grad_f: &mut [Number]) -> bool {
        for (g, &val) in grad_f.iter_mut().zip(x.iter()) {
            *g = val - 1.0;
        }
        true
    }
}

impl NewtonProblem for NLP {
    fn num_hessian_non_zeros(&self) -> usize {
        self.num_variables()
    }
    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        // Constant diagonal hessian
        for i in 0..self.num_hessian_non_zeros() {
            rows[i] = i as Index;
            cols[i] = i as Index;
        }
        true
    }
    fn hessian_values(&self, _x: &[Number], vals: &mut [Number]) -> bool {
        for i in 0..self.num_hessian_non_zeros() {
            vals[i] = 1.0;
        }
        true
    }
}

#[test]
fn quadratic_test() {
    let nlp = NLP {
        z: false,
        iterations: 0,
        x_start: RefCell::new(vec![0.0; 2]),
    };
    let mut ipopt = Ipopt::new_newton(nlp).unwrap();
    ipopt.set_intermediate_callback(Some(NLP::count_iterations_cb));
    ipopt.set_option("tol", 1e-9);
    ipopt.set_option("mu_strategy", "adaptive");
    ipopt.set_option("sb", "yes"); // suppress license message
    ipopt.set_option("print_level", 0); // suppress debug output

    {
        let SolveResult {
            solver_data: SolverDataMut {
                problem, solution, ..
            },
            objective_value: obj,
            status,
            ..
        } = ipopt.solve();

        let x = solution.primal_variables;

        // Ipopt should solve a quadratic problem in one step.
        assert_eq!(status, SolveStatus::SolveSucceeded);
        assert_eq!(problem.iterations, 1);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(obj, 0.0, epsilon = 1e-10);

        problem.save_solution_for_warm_start(solution);
    }

    ipopt.set_option("warm_start_init_point", "yes");

    // Since we have enabled warm start, if we repeat the solve, we should get zero iterations
    // since the problem is already solved.
    {
        let SolveResult {
            solver_data: SolverDataMut { problem, .. },
            status,
            ..
        } = ipopt.solve();
        assert_eq!(status, SolveStatus::SolveSucceeded);
        assert_eq!(problem.iterations, 0);
    }

    // If we add another variable, Ipopt is forced to do another iteration to solve for the
    // additional variable since we dont have warm start information about it.
    ipopt.solver_data_mut().problem.z = true;

    {
        let SolveResult {
            solver_data:
                SolverDataMut {
                    problem,
                    solution:
                        Solution {
                            primal_variables: x,
                            ..
                        },
                },
            objective_value: obj,
            status,
            ..
        } = ipopt.solve();
        assert_eq!(status, SolveStatus::SolveSucceeded);
        assert_eq!(problem.iterations, 1);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], 1.0, epsilon = 1e-10);
        assert_relative_eq!(obj, 0.0, epsilon = 1e-10);
    }
}
