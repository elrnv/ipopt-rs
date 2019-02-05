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
 * This test defines a very basic problem to test Ipopt with.
 * It is a 2 dimensional quadratic that any second order optimizer should be able to solve in one
 * step. In fact here we use Ipopt's implementation of hessian approximation, which is not quite a
 * second order method, although it should easily solve the quadratic problem in one step.
 */
use approx::assert_relative_eq;

use ipopt::*;

struct NLP {
    iterations: usize,
    x_start: Vec<f64>,
}

impl NLP {
    fn count_iterations_cb(&mut self, data: IntermediateCallbackData) -> bool {
        self.iterations = data.iter_count as usize;
        true
    }

    // Save a solution for warm start. Yes this is boilerplate for warm starts. Including this
    // functionality into the crate is future work.
    fn save_solution_for_warm_start(&mut self, solution: Solution) {
        self.x_start.clear();
        self.x_start.extend_from_slice(solution.primal_variables);
    }
}

impl BasicProblem for NLP {
    fn num_variables(&self) -> usize {
        2
    }
    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        x_l.swap_with_slice(vec![-1e20; 2].as_mut_slice());
        x_u.swap_with_slice(vec![1e20; 2].as_mut_slice());
        true
    }
    fn initial_point(&self, x: &mut [Number]) -> bool {
        x.copy_from_slice(&self.x_start);
        true
    }
    fn objective(&self, x: &[Number], obj: &mut Number) -> bool {
        *obj = (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 1.0) * (x[1] - 1.0);
        true
    }
    fn objective_grad(&self, x: &[Number], grad_f: &mut [Number]) -> bool {
        grad_f[0] = 2.0 * (x[0] - 1.0);
        grad_f[1] = 2.0 * (x[1] - 1.0);
        true
    }
}

#[test]
fn quadratic_test() {
    let nlp = NLP {
        iterations: 0,
        x_start: vec![0.0, 0.0],
    };
    let mut ipopt = Ipopt::new_unconstrained(nlp).unwrap();
    ipopt.set_option("tol", 1e-9);
    ipopt.set_option("mu_strategy", "adaptive");
    ipopt.set_option("sb", "yes"); // suppress license message
    ipopt.set_option("print_level", 0); // suppress debug output
    ipopt.set_intermediate_callback(Some(NLP::count_iterations_cb));
    {
        let SolveResult {
            solver_data: SolverDataMut { problem, solution },
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
        assert_eq!(problem.iterations, 0);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(obj, 0.0, epsilon = 1e-10);
    }
}
