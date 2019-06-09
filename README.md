# Steps to use (for Linux)

## 1. Install neper:

Download https://github.com/rquey/neper/archive/3.4.0.tar.gz

Install needed libraries:
```
sudo apt-get install libgsl-dev libnlopt-dev
```

Build Neper:

```
make -j8
```

Check that you can run neper

```
./neper

========================    N   e   p   e   r    =======================
Info   : A software package for polycrystal generation and meshing.
Info   : Version 3.4.0
Info   : Built with: gsl nlopt openmp
Info   : Running on 16 threads.
Info   : <http://neper.sourceforge.net>
Info   : Copyright (C) 2003-2019, and GNU GPL'd, by Romain Quey.
Info   : Comments and bug reports: <neper-users@lists.sourceforge.net>.
========================================================================
```

Add `neper` to `PATH` so you can run it using only `neper` in the terminal. Can also do:

```
sudo ln -s /PATH/TO/NEPER/neper/neper-3.4.0/src/neper /usr/local/bin/neper
```


## 2. Install julia 0.6.

Download https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.4-linux-x86_64.tar.gz.

Extract it.

To run julia, run the executable in `bin/julia`.

## 3. Generate a mesh.

In Julia, run `include("bin/genmesh.jl")` to generate a mesh to `bin/meshes`. Settings can be found in that file (e.g. to control number of grains.)

## 4. Run a simulation

