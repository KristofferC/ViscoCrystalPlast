n_meshes=1
max_rcl=0.4
min_rcl=0.4
n=1
realization=1
NEPER="neper"
dim=3


# rcl 0.400 -> ~ 23 000 elements, 20 grains, 1 000 elements / grain

for i in `seq ${n_meshes} -1 1`; do
    rcl=$(echo "scale = 3; ${min_rcl} * ${i-1} / ${n_meshes} + ${max_rcl} * (1 - ${i-1} / ${n_meshes})" | bc)
    ${NEPER} -M dim${dim}_n${n}_${realization}_gg.tess -dim all -rcl 0${rcl} -format inp
    ${NEPER} -M dim${dim}_n${n}_${realization}_gg.tess -dim all -rcl 0${rcl}
    mv dim${dim}_n${n}_${realization}_gg.inp dim${dim}_n${n}_${realization}_gg_${i}.inp
    mv dim${dim}_n${n}_${realization}_gg.msh dim${dim}_n${n}_${realization}_gg_${i}.msh
    ${NEPER} -V dim${dim}_n${n}_${realization}_gg_${i}.msh -dataelsetcol id -cameraangle 16 -dataelt3dedgerad 0.0015 -dataelt1drad 0.004 -showelt1d all -print dim${dim}_n${n}_${realization}_gg_${i}
done
