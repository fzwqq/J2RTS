#!/bin/bash
#BSUB -J AI2_cp
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
cd ~/uRTS_v2
python3 test_paralell_jin.py --envs-id fullgame-v0,fullgame-v0,fullgame-v0,fullgame-v0,fullgame-v1,fullgame-v1,fullgame-v1,fullgame-v1,fullgame-v2,fullgame-v2,fullgame-v2,fullgame-v2,fullgame-v3,fullgame-v3,fullgame-v3,fullgame-v3 --league None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None --num-process 16 --algo ppo -lr 1e-4  --ai1_socket Jin --ai2_socket Jin --render 0 --saving-prefix multimaps_v2sp
#python3 test_paralell.py  --algo ppo -lr 1e-4 --env-id  fullgame-v1 --render 0 --saving-prefix PPO_League_Six --league Random,WorkerRush,NaiveMCTS,UCT,MCTS,Passive --num-process 16
