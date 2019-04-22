# ipopt-sys

This package provides unsafe Rust bindings to the [Ipopt](https://projects.coin-or.org/Ipopt)
non-linear optimization library.
Unlike most other wrappers for Ipopt, we link against a custom C interface called CNLP, which
mimics Ipopt's own C++ TNLP interface. This serves two purposes:

  1. It helps users who are already familiar with Ipopt's C++ interface to transition into Rust.
  2. It persists the Ipopt solver instance between subsequent solves, which eliminates unnecessary
     additional allocations for initial data and bounds.

This also means that you will need a working C++ compiler and a C++ standard library implementation
available since the CNLP shim currently uses it in the implementation.


## Building

We provide a number of options for building Ipopt from source as well as different methods for retrieving binaries.
Currently supported methods for getting the Ipopt library:

  1. Using pkg-config to find a system installed static or dynamic library.
  2. Manually check system lib directories for a dynamic library.
  3. Build Ipopt from source. You will need fortran compiler libs installed (e.g. `libgfortran` on
     Linux) and one of the following options for linear solvers:
    a. Linking against MKL. (set `MKLROOT` environment variable to specify a custom MKL installation path or if the system path is not found.)
    b. Building with MUMPS/METIS and linking against a system installed OpenBLAS library (Linux) or
Accelerate framework (macOS). If no system BLAS/LAPACK libraries are found, then the default
netlib implementations will be pulled and built.
  4. Download a prebuilt dynamic Ipopt library from JuliaOpt.

Each of these steps are at various levels of polish and currently tested on Linux and macOS systems
only.

Contributions are welcome!
