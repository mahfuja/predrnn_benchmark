export CUDA_VISIBLE_DEVICES=4
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name benchmark \
    --train_data_paths data/benchmark \
    --valid_data_paths data/benchmark  \
    --save_dir checkpoints/benchmark_predrnn_v2 \
    --gen_frm_dir results/benchmark_predrnn_v2 \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_height 1200 \
    --img_width 1100 \
    --img_channel 1 \
    --input_length 6 \
    --total_length 36 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 50000 \
    --display_interval 500 \
    --test_interval 1000 \
    --snapshot_interval 1000
#    --pretrained_model ./checkpoints/kth_predrnn_v2/kth_model.ckpt