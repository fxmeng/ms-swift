# Build a Qwen3-VL pretraining cached dataset from a custom
# {images, input_ids} dataset, then optionally compute packing groups and
# train with Megatron.
#
# Input row format:
#   {
#     "images":    ["chempile-mlift/chebi_20/chebi_20_82142_0.jpg"],
#     "input_ids": [151652, 151655, ..., 151655, 238095, 263001, 189020, 193762,
#                   151653, 19, 17, 15, 12, 15, 13]
#   }
#
# Image paths are resolved against --image_root.
# The source dataset must be an HF dataset saved via
# `dataset.save_to_disk(...)`; we load it here with `load_from_disk`.

# --- Step 1: encode source dataset into full_encode cache ---
# Tuning tips — check `top` first to see what the bottleneck actually is:
#   * %wa > 0 and %us low            → I/O bound → raise --io_threads (8-16).
#   * %wa ~= 0 and %us ~= 100%       → CPU bound → the image processor is the
#                                       bottleneck. Cap resolution via
#                                       --max_pixels (most impactful), keep
#                                       --io_threads small (2-4), and set
#                                       OMP_NUM_THREADS=1 so each worker
#                                       doesn't spawn BLAS threads on top.
#   * load average >> cores          → too many threads: reduce --num_proc or
#                                       --io_threads.
#   * worker CPU% very uneven in top → batch load-imbalance: reduce
#                                       --batch_size (8-16).
#
# OMP_NUM_THREADS=1 is important: each of the N map workers otherwise forks
# its own BLAS thread pool, which produces huge context-switch overhead.
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python examples/train/cached_dataset/encode_pretrain_vl.py \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --source_dataset /path/to/your/hf_save_to_disk_dir \
    --image_root /fsx/youtu-vl/jiayikuang/data_02111332/vl_images/ \
    --output_dir ./qwen3_vl_pretrain_cached \
    --num_proc 32 \
    --batch_size 16 \
    --io_threads 4 \
    --writer_batch_size 256 \
    --max_pixels 1310720 \
    --min_pixels 4096 \
    --val_ratio 0.01

# Output after Step 1:
#   qwen3_vl_pretrain_cached/
#     train/            — Arrow (input_ids, labels, pixel_values_bytes, discrete_tokens, ...)
#     val/              — Arrow (validation split)
#     cache_meta.json


# --- Step 2 (optional): precompute packing groups from cached train set ---
# This calls the swift export path we extended to support packing-only mode.
swift export \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --cached_dataset ./qwen3_vl_pretrain_cached/train \
    --to_cached_dataset true \
    --full_encode true \
    --packing true \
    --max_length 4096 \
    --output_dir ./qwen3_vl_pretrain_cached/train_packing


# --- Step 3: train with pre-cached data (multi-GPU Megatron) ---
# 8 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=14 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
megatron pt \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --save_safetensors true \
    --cached_dataset './qwen3_vl_pretrain_cached/train' \
    --cached_val_dataset './qwen3_vl_pretrain_cached/val' \
    --cached_packing_dataset './qwen3_vl_pretrain_cached/train_packing' \
    --load_from_cache_file true \
    --moe_permute_fusion true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir megatron_output/Qwen3-VL-30B-A3B-Instruct-pretrain \
    --eval_steps 500 \
    --save_steps 500 \
    --max_length 4096 \
    --packing true \
    --freeze_llm true \
    --freeze_vit true \
    --freeze_aligner false \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 0.2 \
    --attention_backend flash
