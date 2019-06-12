 #!/bin/bash

#PBS -S /bin/bash
#PBS -N ELMo-test
#PBS -l walltime=00:10:00
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

mprof run -o "logs/mprofile_<YYYYMMDDhhmmss>.dat" test_alexnet.py -i tf_data/alexnet.tfrecords -o models/alexnet2 -e 18 -b 100