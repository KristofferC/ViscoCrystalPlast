max_rcl=0.05
min_rcl=0.5
n_meshes=15
dim=2
n=3
neper_path="/home/kristoffer/neper-neutral/build/neper"
${neper_path} -T -n ${n} -dim 2 -morpho @seed
for i in `seq 1 ${n_meshes}`; do
    rcl=$(echo "scale = 3; ${min_rcl} * ${i} / ${n_meshes} + ${max_rcl} * (1 - ${i} / ${n_meshes})" | bc)
    ${neper_path} -M n${n}-idrand.tess -dim all -rcl 0${rcl} -format inp
    mv n${n}-idrand.inp meshes/n${n}_rcl${rcl}.inp
done
rm tmp*.msh
rm tmp*.geo
