#!/bin/sh
#PBS -l select=1
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=05:00:00
#PBS -q preemptable
#PBS -A SDR
#PBS -N packkv_accuracy
#PBS -j oe
source ~/miniconda3/etc/profile.d/conda.sh
conda activate packkv

export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.alcf.anl.gov"

export HF_HOME=/lus/grand/projects/SDR/bojiang/

# Change to the directory where qsub was called
cd $PBS_O_WORKDIR

# Execute the specific command
./nohup_run './eval[accuracy]_run.py' 