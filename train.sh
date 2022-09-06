python run_summarization.py \
    --model_name_or_path MrBananaHuman/kobart-base-v2-summarization \
    --tokenizer_name gogamza/kobart-base-v2 \
    --do_train True\
    --num_train_epochs 3 \
    --train_file {train_file_path} \
    --validation_file {validation_file_path} \
    --output_dir ./hf_results \
    --overwrite_output_dir True\
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps 8 \
    --text_column input \
    --summary_column output \
    --learning_rate 1e-5 \
    --fp16 True \
    --fp16_opt_level 'O3' \
    --fp16_full_eval True

python main.py --epochs 3 --tokenizer gogamza/kobart-base-v2 --fp16 True --batch 18 --gradient_accum 8 --local True
python main.py --mode test --fp16 True --eval_batch 60 --batch 1 --tokenizer gogamza/kobart-base-v2 --gradient_accum 8 --valid True