#!/usr/bin/sh

for d in $(seq 0 9)
do
    for t in gw gwtil_ub gwtil_ub_v2 gwtil_lb
    do
        python mnist_2d_bary.py -d $d -t $t
    done
done