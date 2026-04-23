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
# Storage: the encoder writes raw JPEG/PNG bytes into the ``image_bytes``
# column together with the authoritative ``image_grid_thw``. At training
# time ``CachedEncodedDataset`` re-runs the image_processor with the same
# max_pixels / min_pixels recorded in ``cache_meta.json``, so pixel_values
# are reproducible across runs. Cache size scales with compressed source
# images (~50-150 KB/row), ~20-50x smaller than storing decoded fp16
# tensors would be.
#
# For an even more compact layout, pass --preresize_jpeg_quality 95: the
# encoder resizes each image to its smart_resize target and re-encodes as
# JPEG before storing, skipping the processor's resize step entirely at
# training time. Not bit-exact (JPEG Q95 noise), but faster + smaller.
#
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
# Shard mode (only supported mode):
#   --shard_rows N  (required) slices the source dataset into chunks of N rows
#                   and writes each chunk as a self-contained HF dataset under
#                   {output_dir}/shards/shard-XXXXXX-of-YYYYYY/{train,val}/.
#                   Each shard is materialized via a .tmp → rename atomic
#                   handoff, so a crash leaves at most one uncommitted .tmp
#                   sibling, never a half-finished shard. Re-running the same
#                   command skips shards that already exist and continues from
#                   the first missing one — true resume-from-crash.
#
#   Typical shard_rows: 200k-500k. Larger amortizes map setup cost; smaller
#   keeps the redo-cost after a crash bounded.
#
# Multi-pod / multi-host encoding (shard mode only):
#   Just launch this exact script on N pods/hosts, all pointing at the SAME
#   --output_dir on shared storage (NFS / FSx / Lustre). Coordination is
#   filesystem-only: each shard has a ``shard-XXXXXX.claim/`` directory
#   created atomically via ``os.mkdir``; exactly one pod wins the race and
#   encodes that shard, the rest move on to the next unclaimed one. The
#   winner writes a heartbeat every --claim_stale_seconds/10 into the claim
#   dir. If a pod dies (OOM, eviction, node reboot), its heartbeat stops;
#   after --claim_stale_seconds another pod steals the abandoned shard.
#
#   Required: all pods must use identical --model / --max_length /
#   --max_pixels / --min_pixels / --preresize_jpeg_quality / --shuffle_source
#   / --seed / --shard_rows. Otherwise shards produced by different pods
#   would be semantically incompatible even though they commit to the same
#   output_dir. The cache_meta.json in each shard records these args so you
#   can audit them post-hoc.
#
#   Capacity planning: shards are independent, so N pods ≈ Nx throughput
#   until NFS metadata / source-image-read bandwidth saturates (usually
#   around 8-16 pods for a typical fsx mount).

# Single source of truth for --max_length. MUST be identical across Step 1
# (encode filter), Step 2 (packing bin size), Step 3 (training filter).
# Mismatch between any two of these silently invalidates the packing cache
# and/or drops rows at training time.
MAX_LEN=4096
CACHE_ROOT=/dataspace/qwen3_vl_cached
MODEL_PATH=/huggingface/Qwen/Qwen3-VL-30B-A3B-Instruct

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python examples/train/cached_dataset/encode_pretrain_vl.py \
    --model "${MODEL_PATH}" \
    --source_dataset /dataspace/0314_vl_datasets_3in1_tokens_4k_shuffled/ \
    --image_root /fsx/youtu-vl/jiayikuang/data_02111332/vl_images/ \
    --output_dir "${CACHE_ROOT}" \
    --num_proc 128 \
    --batch_size 16 \
    --io_threads 4 \
    --writer_batch_size 256 \
    --max_pixels 1310720 \
    --min_pixels 4096 \
    --max_shard_size 2GB \
    --save_num_proc 16 \
    --shard_rows 500000 \
    --val_ratio 0.01 \
    --max_length "${MAX_LEN}" \
    --preresize_jpeg_quality 95 \
    --drop_below_min_pixels \
    --shuffle_source
    # --shuffle_source: deterministic global shuffle of the source dataset
    # before shard slicing. Flattens the per-shard size distribution when the
    # source is a concat of multiple sub-datasets with different avg image
    # resolutions (otherwise shard sizes can vary 2-5x). Uses HF datasets
    # index-only shuffling; ~free runtime even on 120M+ rows. Keep --seed
    # fixed across resumable runs so already-committed shards stay valid.
    #
    # --max_length 4096 MUST match the --max_length used at training time
    # (Step 3 below). If it doesn't, ms-swift's training-side `_select_dataset`
    # will re-filter the shard → `dataset_modified = True` → the precomputed
    # packing cache is silently ignored and packing is recomputed at each
    # training launch. Keeping the two in sync lets Step 2's packing arrow
    # actually get used.
    #
    # --preresize_jpeg_quality 95: resize each image to its smart_resize
    # target (derived from --max_pixels / --min_pixels) and re-encode as JPEG
    # at Q95 before storing, instead of keeping original bytes. Net effect
    # on a 120M-row, source=mixed(PNG/JPEG/4K-photos) corpus:
    #   - Cache size: ~5-20x smaller (stores target-res JPEG vs. source-res).
    #   - Encode speed: faster (single decode vs. full processor pass).
    #   - Train dataloader speed: faster (processor skips resize, the single
    #     most expensive step).
    #   - Cost: JPEG re-encode adds ~±1 LSB/channel noise; invisible at Q95,
    #     negligible for pretraining. Drop this flag if you need bit-exact
    #     pixel reproducibility (e.g. for eval metric comparisons against a
    #     non-cached run).
    # Locks the cache to (max_pixels, min_pixels) used here: changing those
    # at training time has no effect because images are already at target size.
    #
    # --drop_below_min_pixels: hard-drop any row with an image whose raw
    # (H * W) < --min_pixels. Upscaling a 1x1 placeholder to 64x64 via
    # smart_resize produces a constant-color patch that wastes training
    # compute and spams the "channel dimension is ambiguous" warning from
    # transformers. This filter keeps the cache free of those pathological
    # samples. Applies regardless of --preresize_jpeg_quality.
    #
    # --max_aspect_ratio defaults to 200 (matches transformers' qwen2_vl
    # MAX_RATIO). Rows containing any image with max(H,W)/min(H,W) > 200
    # are dropped at encode time. This guarantees every cached row will be
    # accepted by the processor at training time — without it, the preresize
    # path can silently write shapes like (28, 19992) into the cache that
    # crash the training dataloader on first ingestion. Set to 0 to disable.

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


# --- Pre-flight: validate cache_meta.json before Step 2 / Step 3 ---
# Guards against three very easy footguns:
#   1. Changing --max_length between encode and packing / training (silently
#      re-filters cached rows at load time → packing cache discarded).
#   2. Changing --model between steps (image_processor defaults differ,
#      pixel_values diverge).
#   3. Stale cache from a previous --max_pixels run being fed into a new
#      training launch.
# We read the *first* shard's cache_meta.json as representative; encoding
# guarantees all shards in a given run share identical cache_meta content.
check_cache_meta() {
    local cache_root="$1"
    local expected_max_len="$2"
    local expected_model="$3"
    shopt -s nullglob
    local shards=( "${cache_root}"/shards/shard-* )
    shopt -u nullglob
    if [ "${#shards[@]}" -eq 0 ]; then
        echo "ERROR: no shards under ${cache_root}/shards. Did Step 1 finish?"
        return 1
    fi
    local meta="${shards[0]}/cache_meta.json"
    if [ ! -f "${meta}" ]; then
        echo "ERROR: missing ${meta} — cache was produced by an older encoder"
        echo "       without cache_meta. Re-encode or manually drop one in."
        return 1
    fi
    python - "${meta}" "${expected_max_len}" "${expected_model}" <<'PY'
import json, sys
meta_path, want_max_len, want_model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(meta_path) as f:
    meta = json.load(f)
errs = []
if meta.get('max_length') != want_max_len:
    errs.append(f"  max_length: cache={meta.get('max_length')} expected={want_max_len}")
if meta.get('model') != want_model:
    errs.append(f"  model: cache={meta.get('model')} expected={want_model}")
if errs:
    print(f"cache_meta mismatch in {meta_path}:")
    for e in errs:
        print(e)
    sys.exit(2)
print(f"cache_meta ok: max_length={meta.get('max_length')} model={meta.get('model')} "
      f"store_mode={meta.get('store_mode')} preresize_jpeg_quality={meta.get('preresize_jpeg_quality')}")
PY
    return $?
}


# --- Step 2 (optional): precompute packing groups from cached train shards ---
# swift export's packing-only mode accepts one cached_dataset per invocation.
# With shard mode you run it once per shard; packing precompute is cheap so
# a simple shell loop is fine.
#
# nullglob makes the loop silently do nothing when Step 1 hasn't produced any
# shards yet (otherwise bash leaves the literal `*` in the pattern and swift
# export gets called with a path that contains `*`).
check_cache_meta "${CACHE_ROOT}" "${MAX_LEN}" "${MODEL_PATH}" || exit 1

shopt -s nullglob
for shard in "${CACHE_ROOT}"/shards/shard-*; do
    # Skip shards that have already been packed in a previous run.
    if [ -d "${shard}/train_packing" ]; then
        echo "[pack] skip already-packed: ${shard}"
        continue
    fi
    # Also skip if this shard's train/ doesn't exist yet (e.g. a .tmp half-
    # written shard left over from a crash). Step 1 will regenerate it next run.
    if [ ! -d "${shard}/train" ]; then
        echo "[pack] skip (no train/): ${shard}"
        continue
    fi
    echo "[pack] packing ${shard}"
    swift export \
        --model "${MODEL_PATH}" \
        --cached_dataset "${shard}/train" \
        --to_cached_dataset true \
        --full_encode true \
        --packing true \
        --max_length "${MAX_LEN}" \
        --output_dir "${shard}/train_packing"
done
shopt -u nullglob


# --- Step 3: train with pre-cached data (multi-GPU Megatron) ---
# Shell glob below expands to the per-shard paths. ms-swift's
# --cached_dataset / --cached_val_dataset / --cached_packing_dataset each
# accept a list of paths and concatenate them at load time, so no final
# merge step is required.
#
# dataloader_num_workers for image_bytes mode (CRITICAL):
# image_bytes caches defer the "JPEG decode → rescale → normalize →
# patchify" pipeline to dataloader workers at training time. On our
# max_pixels=1.3M budget this is ~10-20 ms of CPU per image, single-thread.
# With preresize_jpeg_quality=95 the resize step is already done offline, so
# runtime is closer to ~5-8 ms/image, but still non-trivial when the GPU is
# crunching a batch every 80-120 ms.
#
# Recommended starting point: --dataloader_num_workers 16 for 8 GPUs on a
# single node (2 workers / rank). If `top` shows dataloader workers pegged
# at 100% and GPU util dips below 90%, bump to 24-32. If you see the
# opposite (low worker CPU, GPU fully utilized) you can drop it to save RAM.
# Each worker forks its own image_processor instance (~200 MB resident);
# at 32 workers that's ~6 GB of dataloader-side RAM, trivial on a modern
# training node.
#
# Use `shard-*` (not `*`) so leftover `.tmp` dirs from a prior crash are
# excluded. nullglob makes unmatched patterns expand to nothing, so VAL_SHARDS
# being empty (if --val_ratio=0) is fine.
check_cache_meta "${CACHE_ROOT}" "${MAX_LEN}" "${MODEL_PATH}" || exit 1

shopt -s nullglob
TRAIN_SHARDS=( "${CACHE_ROOT}"/shards/shard-*/train )
VAL_SHARDS=(   "${CACHE_ROOT}"/shards/shard-*/val )
PACK_SHARDS=(  "${CACHE_ROOT}"/shards/shard-*/train_packing )
shopt -u nullglob

echo "TRAIN_SHARDS: ${#TRAIN_SHARDS[@]}"
echo "VAL_SHARDS:   ${#VAL_SHARDS[@]}"
echo "PACK_SHARDS:  ${#PACK_SHARDS[@]}"
if [ "${#TRAIN_SHARDS[@]}" -eq 0 ]; then
    echo "ERROR: no train shards found. Did Step 1 finish?"
    exit 1
fi

# 8 * 80GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
OMP_NUM_THREADS=14 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
megatron pt \
    --model "${MODEL_PATH}" \
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
    --max_length "${MAX_LEN}" \
    --packing true \
    --freeze_llm false \
    --freeze_aligner false \
    --freeze_vit true \
    --dataloader_num_workers 16 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 0.2 \
    --attention_backend flash \
    --tensorboard_dir /tmp/tensorboard_logs
