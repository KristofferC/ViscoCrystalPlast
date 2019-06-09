max_rcl=1.0
n_realizatons=5

dim=3
neper_path="/home/kristoffer/neper/build/neper"
for i in 5 10 20 50; do
  for j in `seq 1 ${n_realizatons}`; do
    #${neper_path} -T -n ${n_polys} -id ${id} -dim ${dim} -morpho "diameq:dirac(1),sphericity:lognormal(0.145,0.03,1-x)"
    echo "$i"
    echo "$j"
  done
done
#${neper_path} -T -n ${n_polys} -id ${id} -dim ${dim} -morpho "diameq:dirac(1),sphericity:lognormal(0.145,0.03,1-x)"
#for i in `seq 1 ${n_meshes}`; do
    #rcl=$(echo "scale = 3; ${min_rcl} * ${i} / ${n_meshes} + ${max_rcl} * (1 - ${i} / ${n_meshes})" | bc)
    #${neper_path} -M n${n_polys}-id${id}.tess -dim all -rcl 0${rcl} -format inp
    #mv n${n_polys}-id${id}.inp meshes/n${n_polys}-id${id}_dim${dim}_rcl${rcl}.inp
#done
rm tmp*.msh
rm tmp*.geo
