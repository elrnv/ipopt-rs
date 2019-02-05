# ipopt-rs

A safe Rust interface to the [Ipopt](https://projects.coin-or.org/Ipopt) non-linear optimization
library.

[![On crates.io](https://img.shields.io/crates/v/ipopt.svg)](https://crates.io/crates/ipopt)
[![On docs.rs](https://docs.rs/ipopt/badge.svg)](https://docs.rs/ipopt/)

From the Ipopt webpage:

> Ipopt (**I**nterior **P**oint **OPT**imizer, pronounced eye-pea-Opt) is a software package
> for large-scale nonlinear optimization. It is designed to find (local) solutions of
> mathematical optimization problems of the from
>
>```verbatim
>    min     f(x)
>    x in R^n
>
>    s.t.       g_L <= g(x) <= g_U
>               x_L <=  x   <= x_U
>```
>
> where `f(x): R^n --> R` is the objective function, and `g(x): R^n --> R^m` are the
> constraint functions. The vectors `g_L` and `g_U` denote the lower and upper bounds
> on the constraints, and the vectors `x_L` and `x_U` are the bounds on the variables
> `x`. The functions `f(x)` and `g(x)` can be nonlinear and nonconvex, but should be
> twice continuously differentiable. Note that equality constraints can be
> formulated in the above formulation by setting the corresponding components of
> `g_L` and `g_U` to the same value.

This crate aims to 
  - Reduce the boilerplate, especially for setting up simple unconstrained problems
  - Maintain flexiblity for advanced use cases
  - Prevent common mistakes when defining optimization problems as early as possible using the type
    system and error checking.

# Examples

Solve a simple unconstrained problem using L-BFGS: minimize `(x - 1)^2 + (y -1)^2`


```rust
use approx::*;
use ipopt::*;

struct NLP {
}

impl BasicProblem for NLP {
    // There are two independent variables: x and y.
    fn num_variables(&self) -> usize {
        2
    }    
    // The variables are unbounded. Any lower bound lower than -10^9 and upper bound higher
    // than 10^9 is treated effectively as infinity. These absolute infinity limits can be
    // changed via the `nlp_lower_bound_inf` and `nlp_upper_bound_inf` Ipopt options.
    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        x_l.swap_with_slice(vec![-1e20; 2].as_mut_slice());
        x_u.swap_with_slice(vec![1e20; 2].as_mut_slice());
        true
    }

    // Set the initial conditions for the solver.
    fn initial_point(&self, x: &mut [Number]) -> bool {
        x.swap_with_slice(vec![0.0, 0.0].as_mut_slice());
        true
    }

    // The objective to be minimized.
    fn objective(&self, x: &[Number], obj: &mut Number) -> bool {
        *obj = (x[0] - 1.0)*(x[0] - 1.0) + (x[1] - 1.0)*(x[1] - 1.0);
        true
    }

    // Objective gradient is used to find a new search direction to find the critical point.
    fn objective_grad(&self, x: &[Number], grad_f: &mut [Number]) -> bool {
        grad_f[0] = 2.0*(x[0] - 1.0);
        grad_f[1] = 2.0*(x[1] - 1.0);
        true
    }
}

fn main() {
    let nlp = NLP { };
    let mut ipopt = Ipopt::new_unconstrained(nlp).unwrap();

    // Set Ipopt specific options here a list of all options is available at
    // https://www.coin-or.org/Ipopt/documentation/node40.html
    ipopt.set_option("tol", 1e-9); // set error tolerance
    ipopt.set_option("print_level", 5); // set the print level (5 is the default)

    let solve_result = ipopt.solve();

    assert_eq!(solve_result.status, SolveStatus::SolveSucceeded);
    assert_relative_eq!(solve_result.objective_value, 0.0, epsilon = 1e-10);
    let solution = solve_result.solver_data.solution;
    assert_relative_eq!(solution.primal_variables[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(solution.primal_variables[1], 1.0, epsilon = 1e-10);
}
```

See the tests for more examples including constrained optimization.


# Notes on the custom C API

This crate wraps Ipopt's C++ interface in a custom C API, which is significantly different than
Ipopt's own C API. As such, this crate resembles the Ipopt's C++ API closer than its included C API.
The motivation for maintaining a custom C API is to allow users to reuse Ipopt instances for
multiple solves. For instance, the standard C API prevents users from modifying their problem
structure and setting custom warm starts between solves without reconstructing solver instances. The
custom C API also reduces performance costs, which can be noticeable when Ipopt is used to solve
smaller subproblems.


# License

This repository is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) with the exception of the following files:

  - IpStdCInterface.cpp
  - IpStdCInterface.h
  - IpStdFInterface.c
  - IpStdInterfaceTNLP.cpp
  - IpStdInterfaceTNLP.hpp

which are licensed under the Eclipse Public License(EPL).

