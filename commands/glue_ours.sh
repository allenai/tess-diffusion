
# GLUE should be run with 128+label_length, where label_length=5.
shared_params="--model_name_or_path roberta-large  --do_train --do_eval --do_predict --max_seq_length 133 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard --overwrite_output_dir --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type linear --beta_schedule squaredcos_improved_ddpm   --warmup_steps 500 --logging_steps 50 --save_steps 1000  --add_t5_tags --max_steps 100000   --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 1000"
shared_params_no_topp="--model_name_or_path roberta-large  --do_train --do_eval --do_predict --max_seq_length 133 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard --overwrite_output_dir --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type linear --beta_schedule squaredcos_improved_ddpm   --warmup_steps 500 --logging_steps 50 --save_steps 1000  --add_t5_tags --max_steps 100000   --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 1000"
shared_params_without_predict="--model_name_or_path roberta-large  --do_train --do_eval --max_seq_length 133 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy steps --save_strategy steps   --report_to tensorboard --overwrite_output_dir --pad_to_max_length  --simplex_value 5 --num_diffusion_steps 5000 --conditional_generation seq2seq  --learning_rate 3e-5 --gradient_accumulation_steps 2 --lr_scheduler_type linear --beta_schedule squaredcos_improved_ddpm  --warmup_steps 500 --logging_steps 50 --save_steps 1000  --add_t5_tags --max_steps 100000   --load_best_model_at_end true --checkpoint_best_model  --greater_is_better true --eval_steps 1000"


BASE_DIR="${LOCAL_DIR}/"
PARAMS_FOR_LOCAL=" --save_total_limit 1"
DEBUG_PARAMS="--eval_steps 2 --num_inference_diffusion_steps 3 --per_device_train_batch_size 12 --max_eval_samples 6"
num_inference_diffusion_steps=10
EVAL_PARAMS="--do_train false  --load_states_in_eval_from_model_path true"

# Test weight decay and iterations => with WD was the best with iterations=10
#DATASET="mrpc"
#python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} 

: '
# Runing this one for all datasets.
# with weight_decay
DATASET="cola"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01
'

# DEBUG
#DATASET="cola"
#python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue/debug"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} ${DEBUG_PARAMS} --self_condition_mix_before_weights true --self_condition "logits_multiply" --save_steps 10 --eval_steps 10 

# Training GLUE with self-conditioning mean for now.
: '
DATASET="cola"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_mean/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_mean" --per_device_train_batch_size 32  --gradient_accumulation_steps 4
'

# Training GLUE with self-conditioning max for now.
: '
DATASET="qnli"
python -m torch.distributed.launch --nproc_per_node 2 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_max/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_max" --per_device_train_batch_size 32  --gradient_accumulation_steps 2
'

# Training GLUE with self-conditioning max for now.
: '
DATASET="qnli"
python -m torch.distributed.launch --nproc_per_node 2 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_addition/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_addition" --per_device_train_batch_size 32  --gradient_accumulation_steps 2
'

# Running from a checkpoint with self-condition mean.
: '
DATASET="rte"
python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/ours_glue_self_condition_mean/"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01_from_40K_checkpoint"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_mean" --per_device_train_batch_size 32  --gradient_accumulation_steps 4 --model_name_or_path  "self_condition_mean/checkpoint-40000/"
'

##############################################################################
# Running GLUE with the self-condition mean with the mix before weights setup
##############################################################################
# DATASET="qqp"
# num_inference_diffusion_steps=10
#python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_wd_0.01"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.01 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 4  --self_condition_mix_before_weights true --resume_from_checkpoint "${LOCAL_DIR}/outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_mnli_steps_10_wd_0.01/checkpoint-12000/"

# NOTE: to run for 4 GPU, modify and remove 4 GPUs. Also remove the copy at the end of the output dir.
# python run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 4  --self_condition_mix_before_weights true 

# Training without wd for small data with checkpoint of 500 steps.
# NOTE: runs on 2 GPUS, laters modify the grad_acc and then remove the 2 GPUS.
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_500_steps"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true --save_steps 500 --eval_steps 500


##############################################################################################################################
# Running GLUE with the self-condition mean with the mix before weights setup for the selected max iterations.
##############################################################################################################################
# ******this is selected*******
# NOTE: to run for 4 GPU, modify and remove 4 GPUs. Also remove the copy at the end of the output dir.
# For larger datasets 
# sst2, mnli, qnli, qqp
# DATASET="qqp"
# num_inference_diffusion_steps=10
# python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 25000 --save_checkpoints_on_s3 --resume_from_checkpoint "${LOCAL_DIR}/outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_qqp_steps_10_no_wd_max_steps_set/checkpoint-23000" 

# *****this is selected.******
# For smaller datasets.
# DATASET="cola" # rte, mrpc, cola, stsb, wnli
# python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 12000 --save_checkpoints_on_s3


# training this for mrpc without self-condition.
# For smaller datasets.
DATASET="mrpc" # rte, mrpc, cola, stsb, wnli
num_inference_diffusion_steps=10
# python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params_no_topp} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/mrpc_no_self_condition_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0   --per_device_train_batch_size 32  --gradient_accumulation_steps 1    --max_steps 12000 --save_checkpoints_on_s3



# this is not selected.
# Running small datasets for maximum 9K.
# DATASET="mrpc" # rte, mrpc, cola, stsb, wnli
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_9k_for_small_data"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 9000 --save_checkpoints_on_s3

# this is not selected.
# Running small datasets for 6K. => was not good.
#DATASET="wnli" # rte, mrpc, cola, stsb, wnli
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_6k_for_small_data"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 6000 --save_checkpoints_on_s3

# this is not selected.
# Running small data on 16K steps.
DATASET="rte" # rte, mrpc, cola, stsb, wnli
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_16k_steps"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 16000 --save_checkpoints_on_s3


###############################################################
# We run cola for a different seed.
DATASET="cola" 
seed=13
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set_seed_"${seed}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 12000 --save_checkpoints_on_s3 --seed  ${seed}

###############################################################
# Running self-condition ablations on sst-2
DATASET="sst2"
num_inference_diffusion_steps=10
max_steps=16000
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results_self_condition_ablations/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_steps_"${max_steps}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps ${max_steps} --save_checkpoints_on_s3  --model_name_or_path roberta-base


#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results_self_condition_ablations/self_cond_logits_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_steps_"${max_steps}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --max_steps ${max_steps} --save_checkpoints_on_s3  --model_name_or_path roberta-base 


#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results_self_condition_ablations/no_self_cond_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_steps_"${max_steps}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --max_steps ${max_steps} --save_checkpoints_on_s3  --model_name_or_path roberta-base 

: '
# run their evals.
output_dir=$BASE_DIR"outputs/paper_experiments/glue_results_self_condition_ablations/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_steps_"${max_steps}
model_path=${output_dir}"/checkpoint-13000"
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir ${output_dir}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps ${max_steps} --save_checkpoints_on_s3  --model_name_or_path ${model_path} ${EVAL_PARAMS}

output_dir=$BASE_DIR"outputs/paper_experiments/glue_results_self_condition_ablations/self_cond_logits_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_steps_"${max_steps}
model_path=${output_dir}"/checkpoint-10000"
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir ${output_dir}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --max_steps ${max_steps} --save_checkpoints_on_s3  --model_name_or_path ${model_path} ${EVAL_PARAMS} 

output_dir=$BASE_DIR"outputs/paper_experiments/glue_results_self_condition_ablations/no_self_cond_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_steps_"${max_steps}
model_path=${output_dir}"/checkpoint-16000"
#python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir ${output_dir}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --max_steps ${max_steps} --save_checkpoints_on_s3  --model_name_or_path ${model_path} ${EVAL_PARAMS}
'
###############################################################

# DBEUG
#python -m torch.distributed.launch --nproc_per_node 2  run_glue.py  --dataset_name ${DATASET} ${shared_params} --output_dir $BASE_DIR"outputs/debug"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 25000 --save_checkpoints_on_s3  --max_steps 6 --save_steps 2 --eval_steps 2 --max_eval_samples 6 



#==============================================================
# train on the whole glue dataset without splitting dev/train.
# For larger datasets 
# sst2, mnli, qnli, qqp
#DATASET="qqp"
#num_inference_diffusion_steps=10
#python -m torch.distributed.launch --nproc_per_node 4  run_glue.py  --dataset_name ${DATASET} ${shared_params_without_predict} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set_all_eval_data"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 25000 --save_checkpoints_on_s3 --split_glue false  --do_predict 


# mnli reversed mode.
DATASET="mnli"
num_inference_diffusion_steps=10
#python -m torch.distributed.launch --nproc_per_node 4  run_glue.py  --dataset_name ${DATASET} ${shared_params_without_predict} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set_all_eval_data_mismatched_set_as_eval"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 25000 --save_checkpoints_on_s3 --split_glue false  --do_predict 


# *****this is selected.******
# For smaller datasets.
DATASET="mrpc" # rte, mrpc, cola, stsb, wnli
# python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params_without_predict} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set_all_eval_data"  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 12000 --save_checkpoints_on_s3 --split_glue false --do_predict


# running the above commands for different seeds 
# ===============================================
DATASET="wnli" # rte, mrpc, cola, stsb, wnli
: '
# for seed in 88 67 183 45   
for seed in 59 51 63 25 30 
do 
   python -m torch.distributed.launch --nproc_per_node 4 run_glue.py  --dataset_name ${DATASET} ${shared_params_without_predict} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set_all_eval_data/seed_"${seed}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 12000 --save_checkpoints_on_s3 --split_glue false --do_predict --seed ${seed} --generate_with_seed true
done 
'

# sst2, mnli, qnli, qqp
DATASET="sst2"
num_inference_diffusion_steps=10
# for seed in 63 25 30 88 67 
for seed in  183 45 59 51 
do 
   python -m torch.distributed.launch --nproc_per_node 4  run_glue.py  --dataset_name ${DATASET} ${shared_params_without_predict} --output_dir $BASE_DIR"outputs/paper_experiments/glue_results/ours_self_condition_mean_mix_before_weights_"${DATASET}"_steps_"${num_inference_diffusion_steps}"_no_wd_max_steps_set_all_eval_data/seed_"${seed}  --num_inference_diffusion_steps ${num_inference_diffusion_steps} ${PARAMS_FOR_LOCAL} --weight_decay 0.0 --self_condition "logits_mean"  --per_device_train_batch_size 32  --gradient_accumulation_steps 1  --self_condition_mix_before_weights true  --max_steps 25000 --save_checkpoints_on_s3 --split_glue false  --do_predict  --seed ${seed} --generate_with_seed true
done 
