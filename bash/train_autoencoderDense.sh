 #!/bin/bash

#PBS -S /bin/bash
#PBS -N train-autoencoder
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=2:gputype=p100
#PBS -q gpuq
#PBS -P randstad
#PBS -M badryoubiidrissi@gmail.com
#PBS -o logs/output_autoencDense.txt
#PBS -e logs/error_autoencDense.txt

# Module load

module load cuda/10.0
module load cudnn/7.4.2

cd /workdir/idrissib/AudioMNIST

source activate aud_interp_gpu

mprof run -o "logs/mprofile_<clement>.dat" train_autoencDense.py -i tf_data/audionet.tfrecords -o models/autoencDense -l tensorboard/autoencDense_lr_0.0005 -b 100 -e 100