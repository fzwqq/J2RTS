#bsub -q gpu_v100 -gpu num=1 < ./jobs_four/job_four_league.sh
bsub  < ./jobs_six/job_six_league.sh
#bsub < ./jobs_six/job_six_league_Qnormal.sh
