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
#SBATCH --job-name=cnnmcE
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_train_cnn_mcE.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/BES/BES3EMCPredict/log_train_cnn_mcE.err
  
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
##fix angle, noweight, sat+nosat=all, mcE 0.1-2.5 GeV, 7x7, res:epoch49,train_loss=epoch49,train_loss=0.14148864468583888,valid_loss=0.14977932418177187, lr=9.403456821698725e-07
python $workpath/train_cnn_mcE.py --cfg 'A0' --useSeed True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_all_7x7/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_all_7x7/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_all_7x7/test.txt --out_name $workpath/model/cnn_mcE_sat_nowe_all_low_7x7.pth
##fix angle, noweight, sat+nosat=all, mcE 0.1-2.5 GeV, res:epoch49,train_loss=0.1381954026210457,valid_loss=0.14124014490038372, lr=9.494105434058912e-07
#python $workpath/train_cnn_mcE.py --cfg 'A0' --useSeed True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_all/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_all/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV_all/test.txt --out_name $workpath/model/cnn_mcE_sat_nowe_all_low.pth
##fix angle, noweight, sat+nosat=all, mcE 2-3 GeV, res:epoch49,train_loss=0.1703867410142576,valid_loss=0.17321876846205347, lr=9.483175389283035e-07
#python $workpath/train_cnn_mcE.py --cfg 'A0' --useSeed True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_2to3GeV_all/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_2to3GeV_all/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_2to3GeV_all/test.txt --out_name $workpath/model/cnn_mcE_sat_nowe_all.pth
##fix angle, weight, mcE 2.3-3 GeV, res:epoch49,train_loss=0.0680145555523575,valid_loss=0.06829707058540777, lr=8.472512143922178e-07
#python $workpath/train_cnn_mcE.py --cfg 'A0' --useSeed True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_weight_forMCE_fixAngle_img_2p3to3GeV/train.txt --valid_file $workpath/dataset/noTof_weight_forMCE_fixAngle_img_2p3to3GeV/valid.txt --test_file $workpath/dataset/noTof_weight_forMCE_fixAngle_img_2p3to3GeV/train.txt --out_name $workpath/model/cnn_mcE_sat_we_seed.pth
##fix angle, weight, mcE 2.3-3 GeV, res:epoch49,train_loss=0.06798191726809093,valid_loss=0.06825436770248526, lr=8.472512143922178e-07
#python $workpath/train_cnn_mcE.py --cfg 'A0' --useSeed True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_weight_forMCE_fixAngle_img_2p3to3GeV/train.txt --valid_file $workpath/dataset/noTof_weight_forMCE_fixAngle_img_2p3to3GeV/valid.txt --test_file $workpath/dataset/noTof_weight_forMCE_fixAngle_img_2p3to3GeV/train.txt --out_name $workpath/model/cnn_mcE_sat_we.pth
##fix angle, no weight, mcE 0.1-2.5 GeV, res:
#python $workpath/train_cnn_mcE.py --cfg 'A0' --useSeed True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_2to3GeV/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_2to3GeV/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_2to3GeV/train.txt --out_name $workpath/model/cnn_mcE_sat.pth
##fix angle, no weight, mcE 0.1-2.5 GeV, res:epoch49,train_loss=0.14623542801037348,valid_loss=0.14849785777153174, lr=9.379123935589148e-07
#python $workpath/train_cnn_mcE.py --cfg 'A0' --useSeed True --E_min 0.2 --E_max 2.0 --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_forMCE_fixAngle_img_0p1to2p5GeV/test.txt --out_name $workpath/model/cnn_mcE.pth
