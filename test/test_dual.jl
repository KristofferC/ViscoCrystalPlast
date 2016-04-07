using JuAFEM
using ViscoCrystalPlast
srand(1234)
nslip = 2
uu = rand(8 + 4 * nslip);
uu[ViscoCrystalPlast.g_dofs(2, 4, 1,2,1)] = 0.0
uu[ViscoCrystalPlast.g_dofs(2, 4, 1,2,2)] = 0.0


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
fev = FEValues(Float64, q_rule, function_space)
dt = 0.1
nslip = 2
ele_matstats = ViscoCrystalPlast.CrystPlastDualQD[ViscoCrystalPlast.CrystPlastDualQD(nslip, Dim{2}) for i = 1:length(points(q_rule))]
temp_matstats = ViscoCrystalPlast.CrystPlastDualQD[ViscoCrystalPlast.CrystPlastDualQD(nslip, Dim{2}) for i = 1:length(points(q_rule))]




fe_u = [zero(Vec{2, Float64}) for i in 1:4]
fe_g = [zeros(Float64, 4) for i in 1:nslip]

fe(field) = ViscoCrystalPlast.intf_dual(field, e_coordinates,
              fev, fe_u, fe_g, dt, ele_matstats, temp_matstats, mp)

f = fe(uu)

dim = 2
K_element = zeros(16,16)
    G = ForwardDiff.workvec_eltype(ForwardDiff.GradientNumber, Float64, Val{16}, Val{16})

    nslip = length(mp.angles)
    fe_uG = Vec{dim, G}[zero(Vec{dim, G}) for i in 1:4]
    fe_gG = Vector{G}[zeros(G, 4) for i in 1:nslip]
    feG(field) = ViscoCrystalPlast.intf_dual(field, e_coordinates,
              fev, fe_uG, fe_gG, dt, ele_matstats, temp_matstats, mp)

  Ke! = ForwardDiff.jacobian(feG, ForwardDiff.AllResults, mutates = true, chunk_size = 16)
   Ke!(K_element, uu)
   print(K_element)

