using JuAFEM




function patch_recovery(neighboring_elements, quadrature_values::)


coords = [0.0 1.0 0.5 1.0 0.0;
          0.0 0.0 0.5 1.0 1.0]

topology =
[1 2 4 5;
2 3 3 3;
3 4 5 1;
]

gp_vals = [0.15, 0.45, 0.85, 0.65]


r = zeros(3)

# Linear
function pol!(r, x, xc)
    @assert length(r) == length(x) + 1
    r[1] = 1.
    for i in 1:length(x)
        r[i+1] = x[i] - xc[i]
    end
    return r
end

pol = 1 + (x-0.5) + (y - 0.5)
p = []
function_space = JuAFEM.Lagrange{2, RefTetrahedron, 1}()
q_rule = QuadratureRule(Dim{2}, RefTetrahedron(), 1)
fe_values = FEValues(Float64, q_rule, function_space)

cnt = 1
for i in 1:size(topology, 2)
    x = coords[:, topology[:, i]]
    x_vec = reinterpret(Vec{2, Float64}, x, (3,))
    reinit!(fe_values, x_vec)
    for q_point in 1:length(points(q_rule))
        v = function_vector_value(fe_values, q_point, x_vec)
        push!(p, evaluate_basis(pol, v[1], v[2]))
        cnt += 1
    end
end

A = sum(p[k] * p[k]' for k in 1:4)
b = sum(p[k] * gp_vals[k] for k in 1:4)

aa = A\b

σ = dot(evaluate_basis(pol, 0.5, 0.5), aa)