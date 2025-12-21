{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww21180\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # exp 1.1\
\
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L2_K32 -K 32 -L 2 -P 2 -H 256 512 256\
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L4_K32 -K 32 -L 4 -P 2 -H 256 512 256\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L8_K32 -K 32 -L 8 -P 2 -H 256 512 256\
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L16_K32 -K 32 -L 16 -P 2 -H 256 512 256\
\
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L2_K64 -K 64 -L 2 -P 2 -H 256 512 256\
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L4_K64 -K 64 -L 4 -P 2 -H 256 512 256\
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L8_K64 -K 64 -L 8 -P 2 -H 256 512 256\
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1_L16_K64 -K 64 -L 16 -P 2 -H 256 512 256\
\
}