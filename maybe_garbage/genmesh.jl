rcl = 0.6
n_realizatons = 1
dim = 3
neper_path="/home/kristoffer/neper/build/neper"
n_polys = [3]

for n in n_polys
  for r in 1:n_realizatons
    filename = "n$n-id$r"
    run(`$neper_path -T -n $n -id $r -dim $dim -morpho "diameq:dirac(1),sphericity:lognormal(0.145,0.03,1-x)"`)
    run(`$neper_path -M n$n-id$r.tess -dim all -rcl $rcl -format inp`)
    mv(joinpath(@__DIR__, filename * ".tess"), joinpath(@__DIR__, "meshes", filename * ".tess"); remove_destination=true)
    mv(joinpath(@__DIR__, filename * ".inp" ), joinpath(@__DIR__, "meshes", filename * ".inp"); remove_destination=true)
  end
end
