import os

for i in range(1, 3):
    os.system('CUDA_VISIBLE_DEVICES=0 python greedy_search.py --iter %d --test' % i)
    os.system('CUDA_VISIBLE_DEVICES=0 python greedy_search.py --iter %d' % i)
    os.system('CUDA_VISIBLE_DEVICES=0 python sl_training.py --iter %d' % i)
    os.system('CUDA_VISIBLE_DEVICES=0 python sl_training.py --iter %d' % i)
