# Ipopt-rs

This crate provides a safe Rust interface to the [Ipopt](https://projects.coin-or.org/Ipopt)
non-linear optimization library. From the Ipopt webpage:

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

This crate somewhat simplifies the C-interface exposed by Ipopt. Notably it handles the
boilerplate code required to solve simple unconstrained problems.

# Examples

Solve a simple unconstrained problem using L-BFGS: minimize `(x - 1)^2 + (y -1)^2`


```rust
extern crate ipopt;
#[macro_use] extern crate approx; // for floating point equality tests

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
    fn bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (vec![-1e20; 2], vec![1e20; 2])
    }

    // Set the initial conditions for the solver.
    fn initial_point(&self) -> Vec<Number> {
        vec![0.0, 0.0]
    }

    // The objective to be minimized.
    fn objective(&mut self, x: &[Number], obj: &mut Number) -> bool {
        *obj = (x[0] - 1.0)*(x[0] - 1.0) + (x[1] - 1.0)*(x[1] - 1.0);
        true
    }

    // Objective gradient is used to find a new search direction to find the critical point.
    fn objective_grad(&mut self, x: &[Number], grad_f: &mut [Number]) -> bool {
        grad_f[0] = 2.0*(x[0] - 1.0);
        grad_f[1] = 2.0*(x[1] - 1.0);
        true
    }
}

fn main() {
    let nlp = NLP { };
    let mut ipopt = Ipopt::new_unconstrained(nlp);

    // Set Ipopt specific options here a list of all options is available at
    // https://www.coin-or.org/Ipopt/documentation/node40.html
    ipopt.set_option("tol", 1e-9); // set error tolerance
    ipopt.set_option("print_level", 5); // set the print level (5 is the default)

    let (r, obj) = ipopt.solve();

    {
        let x = ipopt.solution(); // retrieve the solution
        assert_eq!(r, ReturnStatus::SolveSucceeded);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(obj, 0.0, epsilon = 1e-10);
    }
}
```

See the tests for more examples including constrained optimization.

# License

The code within this repository is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
