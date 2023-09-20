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
#SBATCH --job-name=tf_de
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_train_TF_denoise.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_train_TF_denoise.err
  
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
##res:epoch50,train_loss=0.02689562696138933,valid_loss=0.02955779883920487, lr=2.0046979668752146e-09
python $workpath/train_TF_mcE.py --notime False  --useAll True --emb_dim 64 --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/Denoise/train.txt --valid_file $workpath/dataset/Denoise/valid.txt --test_file $workpath/dataset/Denoise/train.txt --out_name $workpath/model/tf_denoise_addTime.pth
##res:epoch50,train_loss=0.028053459134588385,valid_loss=0.02989974011838969, lr=2.0046979668752146e-09
#python $workpath/train_TF_mcE.py  --useAll True --emb_dim 64 --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/Denoise/train.txt --valid_file $workpath/dataset/Denoise/valid.txt --test_file $workpath/dataset/Denoise/train.txt --out_name $workpath/model/tf_denoise.pth
