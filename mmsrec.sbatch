#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=mmsrec_beauty_ngcf-2
#SBATCH --mail-type=END
#SBATCH --mail-user=yw5326@nyu.edu
#SBATCH --output=mmsrec_amazon_beauty_ngcf-2.out

module purge

singularity exec --nv \
	    --overlay /scratch/yw5326/new_env/overlay-25GB-500K.ext3:rw \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash \
        -c "source /ext3/env.sh; conda activate mmsrec; cd /scratch/yw5326/MMSRec; sh finetune_amazon.sh"