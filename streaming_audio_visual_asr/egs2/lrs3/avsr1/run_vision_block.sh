#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test dev"


asr_tag=avsr_contextual_block_transformer
asr_config=conf/avsr_contextual_block_transformer.yaml
lm_config=conf/train_lm.yaml  # Not Used, as use_lm=false

# Audio Frame rate : 16000
# Video Frame rate : 25

export CUDA_VISIBLE_DEVICES=2

./avsr.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --stage 1 \
    --lang en \
    --nj 32 \
    --inference_nj 32 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --lm_config ${lm_config} \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@" \
    --use_streaming true \
    --gpu_inference false \
    --audio_visual true \
    --vis_step 12800 \
    --mouth_roi false \
    # --speed_perturb_factors "0.9 1.0 1.1" \
    # --ngpu 4 \
set -e
set -u
set -o pipefail

