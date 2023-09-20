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
#SBATCH --job-name=testcnnmcE
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_test_cnn_useAll_mcE.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_test_cnn_useAll_mcE.err
  
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
python $workpath/train_cnn_mcE.py --useAll True --iSeed 3 --cfg 'A0'  --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/cnn_useAll_mcE_low_7x7_epoch49.pth --train_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_nosat_7x7/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_nosat_7x7/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_nosat_7x7/train.txt --outFile $workpath/Pred/cnn_train_useAll_mcE_low_7x7.h5
python $workpath/train_cnn_mcE.py --useAll True --iSeed 3 --cfg 'A0'  --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/cnn_useAll_mcE_low_7x7_epoch49.pth --train_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_nosat_7x7/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_nosat_7x7/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_nosat_7x7/test.txt --outFile $workpath/Pred/cnn_test_useAll_mcE_low_7x7.h5
