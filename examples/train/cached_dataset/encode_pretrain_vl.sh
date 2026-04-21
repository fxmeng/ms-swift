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
# Shard mode (recommended for multi-hour runs):
#   --shard_rows N  slices the source dataset into chunks of N rows and writes
#                   each chunk as a self-contained HF dataset under
#                   {output_dir}/shards/shard-XXXXXX-of-YYYYYY/{train,val}/.
#                   Each shard is materialized via a .tmp → rename atomic
#                   handoff, so a crash leaves at most one uncommitted .tmp
#                   sibling, never a half-finished shard. Re-running the same
#                   command skips shards that already exist and continues from
#                   the first missing one — true resume-from-crash.
#
#   Typical shard_rows: 200k-500k. Larger amortizes map setup cost; smaller
#   keeps the redo-cost after a crash bounded.
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python examples/train/cached_dataset/encode_pretrain_vl.py \
    --model /huggingface/Qwen/Qwen3-VL-30B-A3B-Instruct \ \
    --source_dataset /dataspace/0314_vl_datasets_3in1_tokens_4k_shuffled/ \
    --image_root /fsx/youtu-vl/jiayikuang/data_02111332/vl_images/ \
    --output_dir /dataspace/qwen3_vl_cached \
    --num_proc 128 \
    --batch_size 16 \
    --io_threads 4 \
    --writer_batch_size 256 \
    --max_pixels 1310720 \
    --min_pixels 4096 \
    --max_shard_size 2GB \
    --save_num_proc 16 \
    --shard_rows 500000 \
    --val_ratio 0.01

# Output after Step 1 (shard mode):
#   qwen3_vl_pretrain_cached/
#     shards/
#       shard-000000-of-000240/
#         train/   ← Arrow (input_ids, labels, pixel_values_bytes, discrete_tokens, ...)
#         val/     ← Arrow (validation split for this shard, if --val_ratio > 0)
#       shard-000001-of-000240/
#         ...
#       ...
#     cache_meta.json


# --- Step 2 (optional): precompute packing groups from cached train shards ---
# swift export's packing-only mode accepts one cached_dataset per invocation.
# With shard mode you run it once per shard; in practice, packing precompute
# is cheap so this is fine to do in a simple shell loop.
for shard in /dataspace/qwen3_vl_cached/shards/*; do
    if [ -d "${shard}/train" ]; then
        continue
    fi
    swift export \
        --model /huggingface/Qwen/Qwen3-VL-30B-A3B-Instruct \
        --cached_dataset "${shard}/train" \
        --to_cached_dataset true \
        --full_encode true \
        --packing true \
        --max_length 4096 \
        --output_dir "${shard}/train_packing"
done


# --- Step 3: train with pre-cached data (multi-GPU Megatron) ---
# Shell glob below expands to the per-shard paths. ms-swift's
# --cached_dataset / --cached_val_dataset / --cached_packing_dataset each
# accept a list of paths and concatenate them at load time, so no final
# merge step is required.
TRAIN_SHARDS=( /dataspace/qwen3_vl_cached/shards/shard-*/train )
VAL_SHARDS=(   /dataspace/qwen3_vl_cached/shards/shard-*/val )
PACK_SHARDS=(  /dataspace/qwen3_vl_cached/shards/shard-*/train_packing )

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
    --cached_dataset         "${TRAIN_SHARDS[@]}" \
    --cached_val_dataset     "${VAL_SHARDS[@]}" \
    --cached_packing_dataset "${PACK_SHARDS[@]}" \
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
