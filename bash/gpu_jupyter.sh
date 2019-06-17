 #!/bin/bash

qsub -I -N jupyter_gpu -l walltime=00:10:00 -l select=1:ncpus=2:gputype=k40m -q gpuq -P randstad

# Module load

module load cuda/10.0
module load cudnn/7.4.2

cd /workdir/idrissib/AudioMNIST

conda activate aud_interp_gpu

jupyter lab --no-browser --port 9042

#ssh -L localhost:9999:localhost:9999 idrissib@fusion-gpu06

