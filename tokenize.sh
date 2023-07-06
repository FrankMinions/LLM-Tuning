CUDA_VISIBLE_DEVICES=0,1 python tokenize_dataset_rows.py \
    --model_checkpoint THUDM/chatglm2-6b \
    --input_file sentiment_comp_ie.json \
    --instruct_key instruction \
    --prompt_key input \
    --target_key output \
    --save_name sentiment_comp_ie_chatglm2 \
    --max_seq_length 1024 \
    --skip_overlength False
