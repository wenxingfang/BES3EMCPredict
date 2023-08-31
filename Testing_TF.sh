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
#SBATCH --job-name=testtfseed
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_test_TF.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_test_TF.err
  
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
python $workpath/train_TF.py --E_min 0.8 --E_max 1.2 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/tf_notof_0p8_1p2_epoch49.pth --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/train.txt --outFile $workpath/Pred/tf_train_notof_0p8_1p2.h5
python $workpath/train_TF.py --E_min 0.8 --E_max 1.2 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/tf_notof_0p8_1p2_epoch49.pth --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --outFile $workpath/Pred/tf_test_notof_0p8_1p2.h5
#python $workpath/train_TF.py --E_min 0.5 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/tf_notof_Emin0p5_epoch49.pth --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/train.txt --outFile $workpath/Pred/tf_train_notof_Emin0p5.h5
#python $workpath/train_TF.py --E_min 0.5 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/tf_notof_Emin0p5_epoch49.pth --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --outFile $workpath/Pred/tf_test_notof_Emin0p5.h5
#python $workpath/train_TF.py --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/tf_notof_epoch49.pth --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/train.txt --outFile $workpath/Pred/tf_train_notof.h5
#python $workpath/train_TF.py --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/tf_notof_epoch49.pth --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --outFile $workpath/Pred/tf_test_notof.h5
#python $workpath/train_TF.py --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/tf_epoch49.pth --train_file $workpath/dataset/train.txt --valid_file $workpath/dataset/valid.txt --test_file $workpath/dataset/train.txt --outFile $workpath/Pred/tf_train.h5
#python $workpath/train_TF.py --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/tf_epoch49.pth --train_file $workpath/dataset/train.txt --valid_file $workpath/dataset/valid.txt --test_file $workpath/dataset/test.txt --outFile $workpath/Pred/tf_test.h5
