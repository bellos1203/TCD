CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --arch resnet34 --num_segments 8 --seed 1000 --wandb \
        --gd 20 --lr 1e-3 --lr_steps 20 30 --epochs 50 --training --testing \
        --train_batch-size 32 --test_batch-size 16 --exemplar_batch-size 16 -j 8 \
        --start_task 0 --exp 0 --exemplar --nme \
        --test_crops 5 --loss_type nll --store_frames uniform \
        --cl_type DIST --cl_method OURS --init_task 51 --nb_class 2 --K 5 \
        --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --use_importance --t_div --lambda_2 1e-3 --cbf

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --arch resnet34 --num_segments 8 --seed 1993 --wandb \
        --gd 20 --lr 1e-3 --lr_steps 20 30 --epochs 50 --training --testing \
        --train_batch-size 32 --test_batch-size 16 --exemplar_batch-size 16 -j 8 \
        --start_task 0 --exp 1 --exemplar --nme \
        --test_crops 5 --loss_type nll --store_frames uniform \
        --cl_type DIST --cl_method OURS --init_task 51 --nb_class 2 --K 5 \
        --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --use_importance --t_div --lambda_2 1e-3 --cbf

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --arch resnet34 --num_segments 8 --seed 2021 --wandb \
        --gd 20 --lr 1e-3 --lr_steps 20 30 --epochs 50 --training --testing \
        --train_batch-size 32 --test_batch-size 16 --exemplar_batch-size 16 -j 8 \
        --start_task 0 --exp 2 --exemplar --nme \
        --test_crops 5 --loss_type nll --store_frames uniform \
        --cl_type DIST --cl_method OURS --init_task 51 --nb_class 2 --K 5 \
        --lambda_0 1e-2 --lambda_0_type geo --budget_type class --lambda_1 1.0 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --use_importance --t_div --lambda_2 1e-3 --cbf


