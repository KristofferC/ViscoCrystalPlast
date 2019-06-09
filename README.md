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

## 3. Install relevant julia packages

The package manager in julia 0.6 is slow so this will take a while.

```jl
Pkg.add("BlockArrays")  
Pkg.add("DataFrames") 
Pkg.add("JLD")      
Pkg.add("Parameters")                  
Pkg.add("Quaternions")                 
Pkg.add("Tensors")                   
Pkg.add("TimerOutputs")               
Pkg.add("WriteVTK")                    
Pkg.clone("https://github.com/KristofferC/JuAFEM.jl.git")
Pkg.checkout("JuAFEM", "kc/fix_dofs")
```

The exakt versions I used are:

```
julia> Pkg.status()
8 required packages:
 - BlockArrays                   0.4.1
 - DataFrames                    0.11.7
 - JLD2                          0.0.6
 - Parameters                    0.9.2
 - Quaternions                   0.3.1
 - Tensors                       0.7.4
 - TimerOutputs                  0.3.1
 - WriteVTK                      0.8.1
34 additional packages:
 - BinDeps                       0.8.10
 - BinaryProvider                0.3.3
 - Calculus                      0.4.1
 - CategoricalArrays             0.3.13
 - CodecZlib                     0.4.4
 - CommonSubexpressions          0.1.0
 - Compat                        2.1.0
 - Crayons                       0.4.1
 - DataStreams                   0.3.8
 - DataStructures                0.8.4
 - DiffResults                   0.0.4
 - DiffRules                     0.0.10
 - DualNumbers                   0.4.0
 - FileIO                        0.9.1
 - ForwardDiff                   0.7.5
 - JSON                          0.17.2
 - JuAFEM                        0.0.0-             kc/fix_dofs (unregistered)
 - LightXML                      0.7.0
 - Missings                      0.2.10
 - NaNMath                       0.3.2
 - NamedTuples                   4.0.2
 - Nullables                     0.0.8
 - Reexport                      0.1.0
 - Requires                      0.4.4
 - SHA                           0.5.7
 - SIMD                          1.0.0
 - SortingAlgorithms             0.2.1
 - SpecialFunctions              0.6.0
 - StaticArrays                  0.7.2
 - StatsBase                     0.23.1
 - TranscodingStreams            0.5.4
 - URIParser                     0.3.1
 - WeakRefStrings                0.4.7
 ```

## 3. Generate a mesh.

In Julia, run `include("bin/genmesh.jl")` to generate a mesh to `bin/meshes`. Settings can be found in that file (e.g. to control number of grains.). There are some meshes includes in the `bin/meshes` folder already.

## 4. Run a simulation

Run `include("bin/run_simulations.jl"). Different settings available in 

## 5. Analyze results

VTK files that can be visualized in Paraview are stored into `bin/vtks`.

To check saved numerical result, run e.g.

```
using JLD2

load(FileIO.File(format"JLD2", "bin/raw_data/shear_DN_3_dual_Neumann_nslips_12_ngrains_5_microhard.jld2"))
```

which gives a dictionary of stored results. See the notebook in `bin/notebooks` for possible analysis.