include("homogenization.jl")

using Tensors
using JuAFEM
import Homogenization

dbc_f = x -> (x ≈ Vec{3}((-1,-1,-1)) || x ≈ Vec{3}((1,1,1)))  ? true : false

grid = Homogenization.compute_grid(5, dbc_f)
dh = Homogenization.create_dofhandler(grid)
cellvalues, facevalues = Homogenization.create_interpolations()

# Boundary conditions
eps = basevec(Vec{Homogenization.DIM}, 1) ⊗ basevec(Vec{Homogenization.DIM}, 2)
ɛ_bar = 0.05 * eps
E = Homogenization.create_stiffness();
dbc = Homogenization.create_boundaryconditions(grid, dh, (x,t) -> ɛ_bar ⋅ x)

u = 0.05 * rand(ndofs_per_cell(dh))
σ_bar = 1e7 * rand(9)
uσ_bar = [u; σ_bar]
K = create_sparsity_pattern(dh);

f, C, residual_u = Homogenization.doassemble!(cellvalues, facevalues, ɛ_bar, u, σ_bar, K, grid, dh, E)


lhs = [K      C
       C' zeros(9,9)]
rhs   = [zeros(f); -tovoigt(ɛ_bar)];
apply!(lhs, rhs, dbc)

u_σ   = lhs \ rhs;
sigma = u_σ[end-8:end]
uu    = u_σ[1:end-9];


ɛ_list = fill(zero(SymmetricTensor{2,3}), getncells(grid));
σ_list = fill(zero(SymmetricTensor{2,3}), getncells(grid));
ɛ_box, σ_box = Homogenization.check(cellvalues, dh, E, uu, ɛ_list, σ_list)

ɛ_box

σf = reinterpret(Float64, σ_list, (6, getncells(grid)));
ɛf = reinterpret(Float64, ɛ_list, (6, getncells(grid)))

@assert sum(ɛ_list)[1,2] ≈ 31.250000000000313

# Save file
vtkfile = vtk_grid("homo2", dh, uu)
vtk_cell_data(vtkfile, ɛf, "strain")
vtk_cell_data(vtkfile, σf, "stress")
vtk_save(vtkfile)



# Hessian
#=
hess_res = DiffBase.HessianResult(uσ_bar)
hess_cfg = ForwardDiff.HessianConfig{3}(hess_res, uσ_bar)
f = x -> Homogenization.compute_element_potential!(cellvalues, cellcoords,
                           x, ɛ_bar, E)
ForwardDiff.hessian!(hess_res, f, uσ_bar, hess_cfg)





=#



#ForwardDiff.hessian(x ->

#=
"""
dbc_f = x -> norm(x[1]) ≈ 1 ? true : false
dh = create_dofhandler(grid)


dbc = create_boundaryconditions(grid, dh, (x,t) -> ɛ_bar ⋅ x)

K = create_sparsity_pattern(dh);
"""
=#
