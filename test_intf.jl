using Base.Test
begin
    using ForwardDiff
include("common.jl")
 include("visco_crystal_plast_dof_order_tensor.jl")
 function_space = JuAFEM.Lagrange{2, JuAFEM.Square, 1}()
    q_rule = JuAFEM.get_gaussrule(Dim{2}, JuAFEM.Square(), 2)
    fev = FEValues(Float64, q_rule, function_space)
xx = [0.0, 1.0, 1.0, 0.0]
yy = [0.0, 0.0, 1.0, 1.0]

nslip = 2
mss = CrystPlastMS[CrystPlastMS(nslip, 2) for i = 1:length(JuAFEM.points(q_rule))]

x= [xx yy]'
u = zeros(16)
u_prev = zeros(16)

    E = 200000.0
    ν = 0.3
    m = 2.0
    l = 1.e-2
    H⟂ = 0.1E
    Ho = 0.1E
    C = 10.0^(-3)
    tstar = 1000.0
    angles = [20.0, 40.0]
    mp = CrystPlastMP(E, ν, m, H⟂, Ho, l, tstar, C, angles)

dt = 0.1

ff_opt(uu) = intf_opt(uu, u_prev, x, fev, dt, mss, mp)
#Ke = ForwardDiff.jacobian(ff, u)
uu = rand(16)


const FE_CACHE = ForwardDiff.JacobianCache(Val{16}, Val{8}, 16, Float64)
#display(ForwardDiff.jacobian(ff, uu))
#display(ForwardDiff.jacobian(ff_opt, uu, chunk_size=8, cache=FE_CACHE))
display(intf_opt(uu, u_prev, x, fev, dt, mss, mp))
end

K, f = plani4e(xx, yy, [2, 1, 2], hooke(2, E, 0.3))

@test Ke[1:8, 1:8]  ≈ K
#K, f = plani4e(vec(xx), vec(yy), [2, 1, 2], hooke(2, E, 0.3))
Ke = zeros(16, 16)

uuz = copy(uu)
f = intf_opt(uu, u_prev, x, fev, dt, mss, mp)
h = 1e-6
for i in 1:16
    uuz[i] += h
    fi = intf_opt(uuz, u_prev, x, fev, dt, mss,  mp)
    uuz[i] -= h
    Ke[:, i] = (fi - f) / h
end


#=
julia> Ke = ForwardDiff.jacobian(ff, u)
16x16 Array{Float64,2}:
      1.15385e5   48076.9        …  -12625.7    -6312.87   -6312.87
  48076.9             1.15385e5       6312.87    6312.87   12625.7
 -76923.1         -9615.38           12625.7     6312.87    6312.87
   9615.38        19230.8            12625.7    12625.7     6312.87
 -57692.3        -48076.9             6312.87   12625.7    12625.7
 -48076.9        -57692.3        …  -12625.7   -12625.7    -6312.87
  19230.8          9615.38           -6312.87  -12625.7   -12625.7
  -9615.38       -76923.1            -6312.87   -6312.87  -12625.7
  -8240.87         8240.87           13094.8     6547.39   13094.8
  -8240.87         4120.43           26189.6    13094.8     6547.39
  -4120.43         4120.43       …   13094.8    26189.6    13094.8
  -4120.43         8240.87            6547.39   13094.8    26189.6
 -12625.7         12625.7            17093.8     8546.18   17093.9
 -12625.7          6312.87           34188.2    17093.9     8547.17
  -6312.87         6312.87           17093.9    34189.2    17093.8
  -6312.87        12625.7        …    8547.17   17093.8    34188.2

julia> intf(2*ones(u), u_prev, x, fev, dt, mp)
  -46797.5
    -367.161
   27894.4
   32046.4
   90858.4
   -7833.57
   22274.6
   35352.8
   73074.0
  -36227.4
   25081.0
   32745.3
 -117135.0
   44428.1
   31033.9
   30812.4

=#
