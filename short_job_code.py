# import os
# os.system('pwd')
# os.system('which python')
!python experiments/grid_search_command_scripts/KFRA2ConstantDampingOptimizer.py mnist_logreg  --lr 0.1 --damping 0.1 --random_seed 42 --output_dir ../grid_search  --batch_size 128
# !pwd