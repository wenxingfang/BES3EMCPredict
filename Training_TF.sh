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
#SBATCH --job-name=tfseed
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_train_TF.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_train_TF.err
  
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
## remove evt with tof energy>0, E_min=0.8, E_max=1.2, res:epoch49,train_loss=0.09068676241985903,valid_loss=0.0912724190207278, lr=8.812379298292611e-07
python $workpath/train_TF.py --useSeed False --E_min 0.8 --E_max 1.2 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --out_name $workpath/model/tf_notof_0p8_1p2.pth
## remove evt with tof energy>0, E_min=0.5, res:not that good, epoch49,train_loss=0.18322273398036243,valid_loss=0.1836184947837338, lr=9.54733278057983e-07
#python $workpath/train_TF.py --useSeed False --E_min 0.5 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --out_name $workpath/model/tf_notof_Emin0p5.pth
## remove evt with tof energy>0, res:similar
#python $workpath/train_TF.py --useSeed False --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --out_name $workpath/model/tf_notof.pth
## use 3 layers encoder, res: similar results, train_loss=0.14838967788091098,valid_loss=0.15059103879246988 
#python $workpath/train_TF.py --useSeed False --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/train.txt --valid_file $workpath/dataset/valid.txt --test_file $workpath/dataset/test.txt --out_name $workpath/model/tf_3en.pth
## use seed pos info, res: similar with without seed pos info
#python $workpath/train_TF.py --useSeed True --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/train.txt --valid_file $workpath/dataset/valid.txt --test_file $workpath/dataset/test.txt --out_name $workpath/model/tf_useSeed.pth
#python $workpath/train_TF.py --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/train.txt --valid_file $workpath/dataset/valid.txt --test_file $workpath/dataset/test.txt --out_name $workpath/model/tf.pth
