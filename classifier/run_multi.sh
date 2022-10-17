#!/usr/bin/sh


# for ds in MUTAG BZR COX2 PTC_MR IMDB-BINARY IMDB-MULTI
# do
#   python demo_gnn.py -d $ds --delta_g 10 --robust >> output_gnn
# done
################## classification with gw + ct ###################
# DEBUG
# for ds in MUTAG BZR COX2 PTC_MR IMDB-BINARY IMDB-MULTI
# do
#   python demo_train.py -d $ds -tm cttil -df gwtil_lb
# done

################## classification with gw + ct ###################

# for ds in BZR COX2 MUTAG PTC_MR IMDB-BINARY IMDB-MULTI
# do
#   for tm in sp ct cttil
#   do 
#     for df in gw gw_lb gwtil_lb gwtil_ub
#     do
#       # echo "====================" $ds $tm $df normalize >> output_tmp_n
#       # python demo_train.py -d $ds -tm $tm -df $df --normalize -f >> output_tmp_n
#       echo "====================" $ds $tm $df >> output_tmp
#       python demo_train.py -d $ds -tm $tm -df $df -f >> output_tmp
#     done
#   done
# done

################## classification with graph kernel ###################

# for ds in BZR COX2 MUTAG PTC_MR IMDB-BINARY IMDB-MULTI
# do
#   for k in sp rw gk wl
#   # for k in sp
#   do
#     # python demo_grakel.py -d $ds -k $k >> output_grakel
#     python demo_grakel.py -d $ds -k $k
#   done
# done

################## debug ###################
for ds in BZR COX2 MUTAG PTC_MR IMDB-BINARY IMDB-MULTI
do
  for df in gwtil_o
  do
    python demo_train.py -d $ds -tm sp -df $df >> output_svm
  done
done