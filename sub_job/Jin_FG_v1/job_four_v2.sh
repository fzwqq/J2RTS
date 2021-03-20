#!/bin/bash
#BSUB -J Pp_SpCp_Jv0
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu "num=1:mode=exclusive_process:aff=yes"
JAVA_HOME=/seu_share/home/weiweiwu/wlocal/jdk/jdk-11.0.1
#JRE_HOME=${JAVA_HOME}/jre
CLASSPATH=.:${JAVA_HOME}/lib
PATH=$JAVA_HOME/bin:$PATH
PATH=/seu_share/home/weiweiwu/anaconda3/bin:$PATH
LD_LIBRARY_PATH=/seu_share/home/weiweiwu/anaconda3/lib:$LD_LIBRARY_PATH
export JAVA_HOME  CLASSPATH PATH 
cd ~/uRTS_v2
python3 test_jin.py  --algo ppo -lr 1e-5 --env-id  fullgame-v0 --render 0 --saving-prefix Pp_SpCp_Jv0_ --ai1_socket Jin --ai2_socket Jin

