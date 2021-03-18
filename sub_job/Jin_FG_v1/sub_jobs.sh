CUDA_VISIBLE_DEVICES=2 
bsub -q gpu_v100 -gpu num=1 < job_six_v1.sh
bsub -q gpu_v100 -gpu num=1 < job_six_v2.sh
bsub -q gpu_v100 -gpu num=1 < job_six_v3.sh
bsub -q gpu_v100 -gpu num=1 < job_six_league.sh
bsub -q gpu_v100 -gpu num=1 < job_six_league_copy.sh


