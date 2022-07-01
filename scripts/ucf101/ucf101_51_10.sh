CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --seed 1000 --exp 0 \
        --training --testing \
        --start_task 0 --init_task 51 --nb_class 10 \
        --exemplar --nme --K 5 --budget_type class \
        --store_frames uniform --cl_type DIST --cl_method OURS \
        --lambda_0 1e-2 --lambda_0_type geo  --lambda_1 0.3 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --use_importance --t_div --lambda_2 0.01 --cbf

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --seed 1993 --exp 1 \
        --training --testing \
        --start_task 0 --init_task 51 --nb_class 10 \
        --exemplar --nme --K 5 --budget_type class \
        --store_frames uniform --cl_type DIST --cl_method OURS \
        --lambda_0 1e-2 --lambda_0_type geo  --lambda_1 0.3 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --use_importance --t_div --lambda_2 0.01 --cbf

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --seed 2021 --exp 2 \
        --training --testing \
        --start_task 0 --init_task 51 --nb_class 10 \
        --exemplar --nme --K 5 --budget_type class \
        --store_frames uniform --cl_type DIST --cl_method OURS \
        --lambda_0 1e-2 --lambda_0_type geo  --lambda_1 0.3 \
        --eta_learnable --sigma_learnable --fc lsc --num_proxy 1 \
        --use_importance --t_div --lambda_2 0.01 --cbf

