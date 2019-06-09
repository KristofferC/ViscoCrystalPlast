#max_rcl=0.05
#min_rcl=0.5
n_meshes=5
dim=3
#n=20
NEPER="neper"
#for n in 20 50 100; do
#    for i in `seq 1 ${n_meshes}`; do
#        ${NEPER} -T -n ${n} -id ${i} -morpho "diameq:dirac(1),sphericity:lognormal(0.145,0.03,1-x)" -o dim${dim}_n${n}_${i}_gg
#        ${NEPER} -V dim${dim}_n${n}_${i}_gg.tess $C  -print dim${dim}_n${n}_${i}_gg
#    done
#done


# rm tmp*.msh
# rm tmp*.geo
