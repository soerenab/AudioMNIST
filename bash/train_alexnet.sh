 #!/bin/bash

#PBS -S /bin/bash
#PBS -N ELMo-test
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=2:gputype=p100
#PBS -q gpuq
#PBS -P randstad
#PBS -M badryoubiidrissi@gmail.com
#PBS -o logs/output.txt
#PBS -e logs/error.txt

# Module load

module load cuda/10.0
module load cudnn/7.4.2

cd /workdir/idrissib/AudioMNIST

source activate aud_interp_gpu

mprof run -o "logs/mprofile_<YYYYMMDDhhmmss>.dat" train_alexnet.py -i tf_data/alexnet.tfrecords -o models/alexnet3 -l tensorboard/alexnet2_lr_0.0005 -b 100 -e 50