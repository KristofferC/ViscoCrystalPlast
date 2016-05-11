using JuAFEM
using ViscoCrystalPlast
srand(1234)
nslip = 2
uu = rand(8 + 4 * nslip);
uu[ViscoCrystalPlast.g_dofs(2, 4, 1,2,1)] = 0.0
uu[ViscoCrystalPlast.g_dofs(2, 4, 1,2,2)] = 0.0
prev_field = similar(uu)

xx = [ 0.05  0.1   0.1  0.05]
yy = [ 0.05  0.05  0.1  0.1  ]
e_coordinates = [xx yy]'

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

fe(field) = ViscoCrystalPlast.intf(field, prev_field, e_coordinates,
              fev, fe_u, fe_g, dt, mss, temp_mss, mp)

f = fe(uu)
print(f)

dim = 2
K_element = zeros(16,16)
G = ForwardDiff.workvec_eltype(ForwardDiff.GradientNumber, Float64, Val{16}, Val{16})

nslip = length(mp.angles)

feg(field) = ViscoCrystalPlast.intf(field, prev_field, e_coordinates,
              fev, fe_uG, fe_gG, dt, mss, temp_mss, mp)

fe_uG = Vec{dim, G}[zero(Vec{dim, G}) for i in 1:4]
fe_gG = Vector{G}[zeros(G, 4) for i in 1:nslip]


Ke! = ForwardDiff.jacobian(feg, mutates = true, chunk_size = 16)
#@time for i in 1:10^3 Ke!(K_element, uu) end
 Ke!(K_element, uu)
   print(K_element)


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