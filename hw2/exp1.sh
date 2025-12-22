# exp 1.1

#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K 32 -L 2 -P 2 -H 256 512 256
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K 32 -L 4 -P 2 -H 256 512 256
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K 32 -L 8 -P 2 -H 256 512 256
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K 32 -L 16 -P 4 -H 256 512 256

#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K 64 -L 2 -P 2 -H 256 512 256
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K 64 -L 4 -P 2 -H 256 512 256
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K 64 -L 8 -P 2 -H 256 512 256
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K 64 -L 16 -P 4 -H 256 512 256

#exp 1.2

#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K 32 -L 2 -P 2 -H 256 512 256
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K 64 -L 2 -P 2 -H 256 512 256 
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K 128 -L 2 -P 2 -H 256 512 256 

#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K 32 -L 4 -P 2 -H 256 512 256 
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K 64 -L 4 -P 2 -H 256 512 256 
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K 128 -L 4 -P 2 -H 256 512 256 

#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K 32 -L 8 -P 2 -H 256 512 256 
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K 64 -L 8 -P 2 -H 256 512 256 
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K 128 -L 8 -P 2 -H 256 512 256 

#exp 1.3

#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3 -K 64 128 -L 2 -P 2 -H 256 512 256 
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3 -K 64 128 -L 3 -P 2 -H 256 512 256 
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3 -K 64 128 -L 4 -P 2 -H 256 512 256 

#exp 1.4
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 32 -L 8 -P 2 -H 256 512 256 -M resnet 
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 32 -L 16 -P 4 -H 256 512 256 -M resnet 
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 32 -L 32 -P 8 -H 256 512 256 -M resnet 

srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L 2 -P 3 -H 256 512 256 -M resnet 
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L 4 -P 3 -H 256 512 256 -M resnet 
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L 8 -P 6 -H 256 512 256 -M resnet 
