#! /bin/bash
######## Part 1 #########
# Script parameters     #
#########################
  
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
  
# Specify the QOS, mandatory option
#SBATCH --qos=normal
  
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
#SBATCH --account=junogpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=tf
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_train_TF_useAll_mcE.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_train_TF_useAll_mcE.err
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40000
#
# Specify how many GPU cards to us:
#SBATCH --gres=gpu:v100:1
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
  
# list the allocated hosts
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
#hostname
df -h
cd /hpcfs/juno/junogpu/fangwx
source /hpcfs/juno/junogpu/fangwx/setup_conda.sh
#conda activate pytorch2p0
conda activate pyTorch_1p8 
which python
/usr/local/cuda/bin/nvcc --version
export workpath=/hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/
##fix angle, no weight, res:epoch49,train_loss=0.13702093545423655,valid_loss=0.1396712320308524, lr=9.493759059001335e-07
python $workpath/train_TF_mcE.py  --emb_dim 64 --useSeed True  --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_points_0p1to2p5GeV_nosat_7x7/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_points_0p1to2p5GeV_nosat_7x7/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_points_0p1to2p5GeV_nosat_7x7/test.txt --out_name $workpath/model/tf_useSeed_mcE.pth
##fix angle, no weight, res:epoch49,train_loss=0.03445205176916524,valid_loss=0.034733654349580076, lr=9.493759059001335e-07
#python $workpath/train_TF_mcE.py  --emb_dim 64 --useAll True  --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_points_0p1to2p5GeV_nosat_7x7/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_points_0p1to2p5GeV_nosat_7x7/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_points_0p1to2p5GeV_nosat_7x7/test.txt --out_name $workpath/model/tf_useAll_mcE.pth
