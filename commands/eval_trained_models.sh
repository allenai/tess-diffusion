# Evaluation of our models trained on the cloud from a checkpoint

TOP_P=0.5
BASE_DIR=/tmp
# BASE_DIR="/home/"
tokenized_data_path=${BASE_DIR}"/simplex-diffusion/processed_data/openwebtext_256_split_gpt_eval/"
num_inference_diffusion_steps=2500
shared_params="--without_compute_metrics --per_device_train_batch_size 12 --per_device_eval_batch_size 25  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard --overwrite_output_dir --max_seq_length 256  --simplex_value 5 --num_diffusion_steps 5000  --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01  --max_steps 2000000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2   --eval_for_all_metrics  --load_states_in_eval_from_model_path --eval_context_size 25 --skip_special_tokens True"
output_dir=${BASE_DIR}"rabeehk/outputs/paper_experiments/ours_eval/"
truncation_length=56
CHECKPOINT="checkpoint-83000"
DEBUG_PARAMS="--max_eval_samples 2 --num_inference_diffusion_steps 10 --gradient_accumulation_steps 1 "

# On length = 200
# Eval only on one GPU due to speed issue with MAUVE metric.
# our self-condition addition with guidance eval 
# MODEL_PATH=${BASE_DIR}"rabeehk/outputs/paper_experiments/cloudmodels/opentext_ul2_objective_lr_1e-4_length_256_with_self_condition_logits_addition_guidance_5/"${CHECKPOINT} 
# MODEL_NAME="self_condition_logits_addition_guidance_5"
# python -m torch.distributed.launch --nproc_per_node 8  run_mlm.py  --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}   --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} --self_condition logits_addition --guidance_scale 5  
# CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py  --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}   --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} --self_condition logits_addition --guidance_scale 5  


# our self-condition addition 
# MODEL_PATH=${BASE_DIR}"rabeehk/outputs/paper_experiments/cloudmodels/opentext_ul2_objective_lr_1e-4_length_256_with_self_condition_logits_addition/"${CHECKPOINT} 
# MODEL_NAME="self-condition-addition"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} --self_condition logits_addition
# CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} --self_condition logits_addition 


# ul2 model
# MODEL_PATH=${BASE_DIR}"rabeehk/outputs/paper_experiments/cloudmodels/opentext_ul2_objective_lr_1e-4_length_256/"${CHECKPOINT} 
# MODEL_NAME="ul2"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}
# CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}


# original self-condition
# MODEL_PATH=${BASE_DIR}"rabeehk/outputs/paper_experiments/cloudmodels/opentext_ul2_objective_lr_1e-4_length_256_with_self_condition_logits/"${CHECKPOINT} 
# MODEL_NAME="self-condition-original"
# CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 python -m torch.distributed.launch --nproc_per_node 8 --master_port 29510 run_mlm.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}  --self_condition logits 
# CUDA_VISIBLE_DEVICES=8 python compute_mlm_metrics.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}  --self_condition logits


# Running our model for seq_length = 25. ul2
# MODEL_PATH=${BASE_DIR}"rabeehk/outputs/paper_experiments/cloudmodels/opentext_ul2_objective_lr_1e-4_length_256/"${CHECKPOINT} 
# MODEL_NAME="ul2"
# truncation_length=206
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node 4 run_mlm.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}
# CUDA_VISIBLE_DEVICES=0 python compute_mlm_metrics.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH}  --output_dir ${output_dir}/${MODEL_NAME}"_"${TOP_P}"_"${truncation_length}"_"${num_inference_diffusion_steps} --num_inference_diffusion_steps ${num_inference_diffusion_steps}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P}

# For debug
# shared_params="--without_compute_metrics --per_device_train_batch_size 12 --per_device_eval_batch_size 25  --do_eval  --evaluation_strategy steps --eval_steps 1000 --report_to tensorboard  --max_seq_length 256  --simplex_value 5 --num_diffusion_steps 5000  --lr_scheduler_type cosine --learning_rate 1e-4 --pad_to_max_length --beta_schedule squaredcos_improved_ddpm --weight_decay 0.01  --max_steps 2000000 --gradient_accumulation_steps 8 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --conditional_generation ul2   --eval_for_all_metrics  --load_states_in_eval_from_model_path --eval_context_size 25 --skip_special_tokens True"
MODEL_PATH=${BASE_DIR}"rabeehk/outputs/paper_experiments/cloudmodels/opentext_ul2_objective_lr_1e-4_length_256/"${CHECKPOINT} 
MODEL_NAME="ul2"
truncation_length=56
# -m torch.distributed.launch --nproc_per_node 2
python  run_mlm.py --truncation_length ${truncation_length} --do_train --model_name_or_path ${MODEL_PATH} --do_train --output_dir ${output_dir}"/test1" --num_inference_diffusion_steps ${num_inference_diffusion_steps}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} ${DEBUG_PARAMS} --save_steps 20 --eval_steps 2 # --resume_from_checkpoint "${LOCAL_DIR}/outputs/paper_experiments/ours_eval//test/checkpoint-80/"
# python compute_mlm_metrics.py --truncation_length ${truncation_length} --model_name_or_path ${MODEL_PATH} --do_train --output_dir ${output_dir}"/test" --num_inference_diffusion_steps ${num_inference_diffusion_steps}  ${shared_params} --tokenized_data_path ${tokenized_data_path} --top_p ${TOP_P} ${DEBUG_PARAMS}





