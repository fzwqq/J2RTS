#!/bin/bash
#BSUB -J PPO_SP_J6pll
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu num=1
JAVA_HOME=/seu_share/home/weiweiwu/wlocal/jdk/jdk-11.0.1
#JRE_HOME=${JAVA_HOME}/jre
CLASSPATH=.:${JAVA_HOME}/lib
PATH=$JAVA_HOME/bin:$PATH
PATH=/seu_share/home/weiweiwu/anaconda3/bin:$PATH
LD_LIBRARY_PATH=/seu_share/home/weiweiwu/anaconda3/lib:$LD_LIBRARY_PATH
export JAVA_HOME  CLASSPATH PATH 
cd ~/uRTS_v2/microrts
python3 test_paralell.py  --algo ppo -lr 1e-4 --env-id  fullgame-v1 --render 0 --league None --saving-prefix PPO_SP_Six_Paralell --num-process 16
