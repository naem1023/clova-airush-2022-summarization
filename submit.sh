# NSML Training examples

nsml run -g 5 -d {nsml_dataset} -e run.py --esm {your_id} -a '--model_name_or_path hyunwoongko/kobart
    --do_train True
    --num_train_epochs 6
    --dataset_name {Huggingface_dataset_hub_full_name}
    --source_prefix "요약: "
    --output_dir ./bart-v2-x-fp32
    --overwrite_output_dir True
    --per_device_train_batch_size 22 
    --gradient_accumulation_steps 8 
    --text_column input 
    --summary_column output 
    --learning_rate 1e-5 
    --seed 42'

nsml run -g 4 -d {nsml_dataset} -e main.py --esm {your_id} -a "--epochs 8 --max_target_len 256 --lr 1e-4 --model paust/pko-t5-large --fp16 True --batch 2 --gradient_accum 8 --overwrite_cache True"

nsml run -d {nsml_dataset} -e main.py --esm {your_id} -a "--epochs 3 --model paust/pko-t5-large --lr 1e-4 --load_session {full_nsml_session_name} --fp16 True --batch 12 --eval_batch 52 --gradient_accum 8"