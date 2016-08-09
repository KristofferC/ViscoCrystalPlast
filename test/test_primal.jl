using JuAFEM
using ViscoCrystalPlast
using BlockArrays

prim_field = [0.0006000000238418602,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
prev_prim_field = zeros(prim_field)
e_coordinates = [0.0 0.0 0.0192798204158598
               0.030000001192093007 0.0 0.021354516264327765]

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
function_space = JuAFEM.Lagrange{2, RefTetrahedron, 1}()
q_rule = QuadratureRule(Dim{2}, RefTetrahedron(), 1)
n_qpoints = length(points(q_rule))
fe_values = FEValues(Float64, q_rule, function_space)
dt = 0.01
nslip = 2

import ViscoCrystalPlast: create_quadrature_data
mss = [ViscoCrystalPlast.CrystPlastPrimalQD(nslip, Dim{2}) for i = 1:n_qpoints]
temp_mss = [ViscoCrystalPlast.CrystPlastPrimalQD(nslip, Dim{2}) for i = 1:n_qpoints]



primal_prob = ViscoCrystalPlast.PrimalProblem(nslip, function_space)

e_coordsvec = [Vec{2}(e_coordinates[:, i]) for i in 1:3]

fe(field) = ViscoCrystalPlast.intf(primal_prob, field, prev_prim_field, e_coordsvec,
              fe_values, dt, mss, temp_mss, mp)

prim_field = ones(12)
f, K = fe(prim_field)


fe_u = Vec{2, Float64}[zero(Vec{2, Float64}) for i in 1:3]
fe_g = Vector{Float64}[zeros(Float64, 3) for i in 1:nslip]
fe_g2 = Vector{Float64}[zeros(Float64, 3) for i in 1:nslip]

feg2(field) = ViscoCrystalPlast.intf(field, prev_prim_field, e_coordsvec,
              fe_values, fe_u, fe_g, fe_g2, dt, mss, temp_mss, mp)

feg2(prim_field)

dim = 2
K_element = zeros(12,12)
G = ForwardDiff.Dual{12, Float64}

nslip = length(mp.angles)

fe_uG = Vec{dim, G}[zero(Vec{dim, G}) for i in 1:3]
fe_gG = Vector{G}[zeros(G, 3) for i in 1:nslip]
fe_gG2 = Vector{G}[zeros(G, 3) for i in 1:nslip]

feg(field) = ViscoCrystalPlast.intf(field, prev_prim_field, e_coordsvec,
              fe_values, fe_uG, fe_gG, fe_gG2, dt, mss, temp_mss, mp)



Ke! = (K, u) -> ForwardDiff.jacobian!(K, feg, u)

@time for i in 1:10^4
  fe(prim_field)
end

@time for i in 1:10^4
  Ke!(K_element, prim_field)
end

nslips = 2
KB = PseudoBlockArray(K_element, [6, [3 for i in 1:nslips]...],
                                                      [6, [3 for i in 1:nslips]...])

#=
fe_u = [zero(Vec{2, Float64}) for i in 1:3]
fe_g = [zeros(Float64, 3) for i in 1:nslip]


fe(field) = ViscoCrystalPlast.intf(field, prev_prim_field, e_coordinates, fe_values, fe_u, fe_g, dt, ele_matstats, mp)

prim_field = ones(12)
f = fe(prim_field)

f_res =  [-2033.6006392059926,-435.11788263371466,43.57985123971975,43.57985123971975,155.60591531339742,
          1519.3786420098013,43.57985123971975,43.57985123971975,1877.9947238925952,-1084.2607593760868,
          43.57985123972107,43.57985123972107]


@test norm(f - f_res) / norm(f_res) <= 1e-6


=#
