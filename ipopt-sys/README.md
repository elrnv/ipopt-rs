# ipopt-sys

This package provides unsafe Rust bindings to the [Ipopt](https://projects.coin-or.org/Ipopt)
 non-linear optimization library.

This build script builds Ipopt from source. This means you need a BLAS library, a fortran compiler
and a linear solver installed on your system. MKL is supported and tested on macOS.

In addition you will need `libclang`.

It has currently only been tested on macOS.
