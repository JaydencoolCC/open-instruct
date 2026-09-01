export PYTHONPATH=/data/home/zhanghx/code/open-instruct/OLMo-core-main/src

python \
  OLMo-core-main/src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path checkpoints/rl/olmo3-7b-GRPO-20260806_092823__42__1786008555_checkpoints/step_500 \
  --huggingface-output-dir /home/zhanghx/models/rl/step_500 \
 --tokenizer /data/home/zhanghx/code/open-instruct/checkpoints/sft_models/tokenizer \
  --max-sequence-length 32768 \
  --device cpu