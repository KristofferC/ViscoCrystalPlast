max_rcl=2.0
min_rcl=0.6
n_meshes=5
id=3
n_polys=5
dim=3
neper_path="/home/kristoffer/neper/build/neper"
${neper_path} -T -n ${n_polys} -id ${id} -dim ${dim} -morpho "diameq:dirac(1),sphericity:lognormal(0.145,0.03,1-x)"
for i in `seq 1 ${n_meshes}`; do
    rcl=$(echo "scale = 3; ${min_rcl} * ${i} / ${n_meshes} + ${max_rcl} * (1 - ${i} / ${n_meshes})" | bc)
    ${neper_path} -M n${n_polys}-id${id}.tess -dim all -rcl 0${rcl} -format inp
    mv n${n_polys}-id${id}.inp meshes/n${n_polys}-id${id}_dim${dim}_rcl${rcl}.inp
done
rm tmp*.msh
rm tmp*.geo
