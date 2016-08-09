using JuAFEM
using ViscoCrystalPlast
srand(1234)
nslip = 2
uu = rand(8 + 4 * nslip);
uu[ViscoCrystalPlast.compute_γdofs(2, 4, 1,2,1)] = 0.0
uu[ViscoCrystalPlast.compute_γdofs(2, 4, 1,2,2)] = 0.0
prev_field = similar(uu)

xx = [ 0.05  0.1   0.1  0.05]
yy = [ 0.05  0.05  0.1  0.1  ]
e_coordinates = [xx; yy]

function setup_material{dim}(::Type{Dim{dim}})
    E = 200000.0
    ν = 0.3
    n = 2.0
    lα = 0.5
    H⟂ = 0.1E
    Ho = 0.1E
    C = 1.0e3
    tstar = 1000.0
    angles = [20.0, 40.0]
    mp = ViscoCrystalPlast.CrystPlastMP(Dim{dim}, E, ν, n, H⟂, Ho, lα, tstar, C, angles)
    return mp
end

mp = setup_material(Dim{2})
function_space = JuAFEM.Lagrange{2, RefCube, 1}()
q_rule = QuadratureRule(Dim{2}, RefCube(), 2)
n_qpoints = length(points(q_rule))
fev = FEValues(Float64, q_rule, function_space)
dt = 0.1
nslip = 2

import ViscoCrystalPlast: DualProblem, create_quadrature_data
mss = [ViscoCrystalPlast.CrystPlastDualQD(nslip, Dim{2}) for i = 1:n_qpoints]
temp_mss = [ViscoCrystalPlast.CrystPlastDualQD(nslip, Dim{2}) for i = 1:n_qpoints]

fe_u = [zero(Vec{2, Float64}) for i in 1:4]
fe_g = [zeros(Float64, 4) for i in 1:nslip]
fe_g2 = [zeros(Float64, 4) for i in 1:nslip]
dual_prob = ViscoCrystalPlast.DualProblem(nslip, function_space)
fe(field) = ViscoCrystalPlast.intf(dual_prob, field, prev_field, e_coordinates,
              fev, dt, mss, temp_mss, mp)


f, K = fe(uu)

ff = copy(f)
display(f)



K_num = zeros(16, 16)

h = 1e-4
for i in 1:16
  uu[i] += h
  f2, qq = fe(uu)
  K_num[:, i] = (f2 - ff) / h
  uu[i] -= h
end

e_coordsvec = [Vec{2}(e_coordinates[:, i]) for i in 1:3]

fe2(field) = ViscoCrystalPlast.intf_dual(field, prev_field, e_coordinates,
              fev, fe_u, fe_g, fe_g2, dt, mss, temp_mss, mp)



dim = 2
N = 16
K_element = zeros(N,N)
G = ForwardDiff.Dual{N,Float64}

nslip = length(mp.angles)

fe_uG = Vec{dim, G}[zero(Vec{dim, G}) for i in 1:4]
fe_gG = Vector{G}[zeros(G, 4) for i in 1:nslip]
fe_gG2 = Vector{G}[zeros(G, 4) for i in 1:nslip]


feg3(field) = ViscoCrystalPlast.intf_dual(field, prev_field, e_coordinates,
              fev, fe_uG, fe_gG, fe_gG2, dt, mss, temp_mss, mp)




Ke! = (K, u) -> ForwardDiff.jacobian!(K, feg3, u)

Ke!(K_element, uu)
#
##@time for i in 1:10^3 Ke!(K_element, uu) end
#
# #print(K_element[ ViscoCrystalPlast.γ_dofs(2, 4, 1,2,1),ViscoCrystalPlast.u_dofs(2, 4, 1,2)])
#   print(K_element)

#=
r = zeros(2*nslip)
X = rand(2*nslip)
Y = rand(4 + nslip)
R!(r, x) = ViscoCrystalPlast.compute_residual!(r, x, Y, 0.1, mp, mss[1])
J = ForwardDiff.jacobian(R!, output_length = length(X))
J(X)

jbuf = zeros(2*nslip, 2*nslip)
ViscoCrystalPlast.compute_jacobian!(jbuf, X, Y, 0.1, mp, mss[1])


RX!(r, y) = ViscoCrystalPlast.compute_residual!(r, X, y, 0.1, mp, mss[1])
JX = ForwardDiff.jacobian(RX!, output_length = length(X))
JX(Y)

ViscoCrystalPlast.compute_residual!(r, X, Y, Δt, mp, mss)
=#
