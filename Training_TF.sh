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
##fix angle, no weight, res:epoch49,train_loss=0.13781152634361457,valid_loss=0.1381211649584858, lr=9.572201399772092e-07
python $workpath/train_TF_feas.py --usemc True --emb_dim 64 --useSeed True --E_min 0.2 --E_max 2.0 --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_noweight_feas_mc_fixAngle/train.txt --valid_file $workpath/dataset/noTof_noweight_feas_mc_fixAngle/valid.txt --test_file $workpath/dataset/noTof_noweight_feas_mc_fixAngle/test.txt --out_name $workpath/model/tf_feas_fixA_noW.pth
##fix angle, res:epoch49,train_loss=0.14401232957804125,valid_loss=0.14493032897655683, lr=7.664527889774339e-07
#python $workpath/train_TF_feas.py --usemc True --emb_dim 64 --useSeed True --E_min 0.2 --E_max 2.0 --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_flat_feas_mc_fixAngle/train.txt --valid_file $workpath/dataset/noTof_flat_feas_mc_fixAngle/valid.txt --test_file $workpath/dataset/noTof_flat_feas_mc_fixAngle/test.txt --out_name $workpath/model/tf_feas_fixA.pth
##use mc theta, sinphi, cosphi features, res:epoch49,train_loss=0.1637420359481479,valid_loss=0.17131878909269388, lr=8.784470031188706e-07
#python $workpath/train_TF_feas.py --usemc True --emb_dim 64 --useSeed True --E_min 0.2 --E_max 2.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_flat_feas_mc/train.txt --valid_file $workpath/dataset/noTof_flat_feas_mc/valid.txt --test_file $workpath/dataset/noTof_flat_feas_mc/test.txt --out_name $workpath/model/tf_feas_mc.pth
## remove evt with tof energy>0, flat E, E_min=1. E_max=1.5, more features, res:epoch49,train_loss=0.10700160545053111,valid_loss=0.1114613297674395, lr=6.587099502769301e-07
#python $workpath/train_TF_feas.py --emb_dim 64 --useSeed True --E_min 1. --E_max 1.5 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_flat_feas/train.txt --valid_file $workpath/dataset/noTof_flat_feas/valid.txt --test_file $workpath/dataset/noTof_flat_feas/test.txt --out_name $workpath/model/tf_feas_1p0.pth
## remove evt with tof energy>0, flat E, E_min=0.2 E_max=2, more features, more evts, res:epoch49,train_loss=0.1642154111268999,valid_loss=0.16953971689517575, lr=9.310830965453861e-0
#python $workpath/train_TF_feas.py --emb_dim 64 --useSeed True --E_min 0.2 --E_max 2.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_flat_feas/train_all.txt --valid_file $workpath/dataset/noTof_flat_feas/valid.txt --test_file $workpath/dataset/noTof_flat_feas/test.txt --out_name $workpath/model/tf_feas_0p2_all.pth
## remove evt with tof energy>0, flat E, E_min=0.2 E_max=2, more features, helps a bit, res:epoch49,train_loss=0.16609676274947918,valid_loss=0.17218548613317478, lr=8.784470031188706e-0
#python $workpath/train_TF_feas.py --emb_dim 64 --useSeed True --E_min 0.2 --E_max 2.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_flat_feas/train.txt --valid_file $workpath/dataset/noTof_flat_feas/valid.txt --test_file $workpath/dataset/noTof_flat_feas/test.txt --out_name $workpath/model/tf_feas_0p2.pth
## remove evt with tof energy>0, flat E, E_min=0.5, E_max=2, more features, helps a bit, res:epoch49,train_loss=0.16501813294179615,valid_loss=0.17065573315510937, lr=8.692886661436842e-07
#python $workpath/train_TF_feas.py --emb_dim 64 --useSeed True --E_min 0.5 --E_max 2.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_flat_feas/train.txt --valid_file $workpath/dataset/noTof_flat_feas/valid.txt --test_file $workpath/dataset/noTof_flat_feas/test.txt --out_name $workpath/model/tf_feas.pth
## remove evt with tof energy>0, flat E, E_min=0.5, E_max=2, more emb, res:
#python $workpath/train_TF.py --emb_dim 64 --useSeed True --E_min 0.5 --E_max 2.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof_flat/train.txt --valid_file $workpath/dataset/noTof_flat/valid.txt --test_file $workpath/dataset/noTof_flat/test.txt --out_name $workpath/model/tf_flatE.pth
## remove evt with tof energy>0, E_min=0.8, E_max=1.2, more emb, res:epoch49,train_loss=0.09040215136420347,valid_loss=0.09118849320772958, lr=8.812379298292611e-07
#python $workpath/train_TF.py --emb_dim 64 --useSeed True --E_min 0.8 --E_max 1.2 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --out_name $workpath/model/tf_emb64_.pth
## remove evt with tof energy>0, E_min=0.8, E_max=1.2, more dense layer, res:epoch49,train_loss=0.09110269974695255,valid_loss=0.09194205165589037, lr=8.812379298292611e-07
#python $workpath/train_TF.py --useSeed False --E_min 0.8 --E_max 1.2 --notime True --fcs 1024 512 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --out_name $workpath/model/tf_add512_.pth
## remove evt with tof energy>0, E_min=0.8, E_max=1.2, res:epoch49,train_loss=0.09068676241985903,valid_loss=0.0912724190207278, lr=8.812379298292611e-07
#python $workpath/train_TF.py --useSeed False --E_min 0.8 --E_max 1.2 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --out_name $workpath/model/tf_notof_0p8_1p2.pth
## remove evt with tof energy>0, E_min=0.5, res:not that good, epoch49,train_loss=0.18322273398036243,valid_loss=0.1836184947837338, lr=9.54733278057983e-07
#python $workpath/train_TF.py --useSeed False --E_min 0.5 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --out_name $workpath/model/tf_notof_Emin0p5.pth
## remove evt with tof energy>0, res:similar
#python $workpath/train_TF.py --useSeed False --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/noTof/train.txt --valid_file $workpath/dataset/noTof/valid.txt --test_file $workpath/dataset/noTof/test.txt --out_name $workpath/model/tf_notof.pth
## use 3 layers encoder, res: similar results, train_loss=0.14838967788091098,valid_loss=0.15059103879246988 
#python $workpath/train_TF.py --useSeed False --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/train.txt --valid_file $workpath/dataset/valid.txt --test_file $workpath/dataset/test.txt --out_name $workpath/model/tf_3en.pth
## use seed pos info, res: similar with without seed pos info
#python $workpath/train_TF.py --useSeed True --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/train.txt --valid_file $workpath/dataset/valid.txt --test_file $workpath/dataset/test.txt --out_name $workpath/model/tf_useSeed.pth
#python $workpath/train_TF.py --E_min 1.0 --notime True --fcs 1024 128 1 --epochs 50 --lr 5e-4 --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/train.txt --valid_file $workpath/dataset/valid.txt --test_file $workpath/dataset/test.txt --out_name $workpath/model/tf.pth
