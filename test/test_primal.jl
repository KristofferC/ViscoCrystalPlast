
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
fe_values = FEValues(Float64, q_rule, function_space)
dt = 0.01
nslip = 2
ele_matstats = ViscoCrystalPlast.CrystPlastPrimalQD[ViscoCrystalPlast.CrystPlastPrimalQD(nslip, Dim{2}) for i = 1:length(points(q_rule))]




fe_u = [zero(Vec{2, Float64}) for i in 1:3]
fe_g = [zeros(Float64, 3) for i in 1:nslip]


fe(field) = ViscoCrystalPlast.intf_opt(field, prev_prim_field, e_coordinates, fe_values, fe_u, fe_g, dt, ele_matstats, mp)

prim_field = ones(12)
f = fe(prim_field)

f_res =  [-2033.6006392059926,-435.11788263371466,43.57985123971975,43.57985123971975,155.60591531339742,
          1519.3786420098013,43.57985123971975,43.57985123971975,1877.9947238925952,-1084.2607593760868,
          43.57985123972107,43.57985123972107]


@test norm(f - f_res) / norm(f_res) <= 1e-6

