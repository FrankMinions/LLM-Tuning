 CUDA_VISIBLE_DEVICES=0,1,2,3 python baichuan_resume_sft.py \
     --tokenized_dataset sentiment_comp_ie_shuffled_baichuan-7B \
     --lora_rank 8 \
     --lora_target W_pack \
     --per_device_train_batch_size 4 \
     --gradient_accumulation_steps 1 \
     --num_train_epochs 3.0 \
     --save_steps 200 \
     --save_total_limit 3 \
     --learning_rate 1e-4 \
     --fp16 \
     --remove_unused_columns false \
     --logging_steps 10 \
     --output_dir weights/sentiment_comp_ie_shuffled_baichuan-7B

