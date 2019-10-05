export ENV="/home/israel/miniconda3/envs/bertenv/bin/python";
export SCRIPT="/home/israel/Documents/Mestrado/MLMaster/mlpack/scripts/run_glue.py";
export GLUE_DIR="/home/israel/Documents/Mestrado/MLMaster/glue_data";
export TASK_NAME="MRPC"

"${ENV}" "${SCRIPT}" \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-6 \
  --num_train_epochs 8.0 \
  --output_dir /tmp/$TASK_NAME/ \
  --fp16