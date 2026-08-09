export PYTHONPATH=/data/home/zhanghx/code/open-instruct/OLMo-core-main/src

/data/home/zhanghx/.conda/envs/olmo3_sft/bin/python \
  OLMo-core-main/src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path checkpoints/sft_models/step3256 \
  --huggingface-output-dir /data/home/zhanghx/olmo3/models/sft_models/step3256-hf \
 --tokenizer /data/home/zhanghx/code/open-instruct/checkpoints/sft_models/tokenizer \
  --max-sequence-length 32768 \
  --device cpu