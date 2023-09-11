PARAMS_FOR_LOCAL=" --save_total_limit 1 "

#######################
# Run simplification.
#######################
# not used.
#learning_rate=3e-5
#python run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/tune_lrs_simplification/lr_"${learning_rate}"_simplification_baseline" --per_device_train_batch_size=12 --per_device_eval_batch_size=25 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate --resume_from_checkpoint "${LOCAL_DIR}/outputs/paper_experiments/tune_lrs_simplification/lr_3e-5_simplification_baseline/checkpoint-246000/" ${PARAMS_FOR_LOCAL}
# learning_rate=3e-5
# not used.
# python run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/tune_lrs_simplification/lr_"${learning_rate}"_simplification_baseline_with_wd" --per_device_train_batch_size=12 --per_device_eval_batch_size=25 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000    --predict_with_generate  --weight_decay 0.01 --resume_from_checkpoint "${LOCAL_DIR}/outputs/paper_experiments/tune_lrs_simplification/lr_3e-5_simplification_baseline_with_wd/checkpoint-245000/" ${PARAMS_FOR_LOCAL}
# lr=3e-5 is the best and also without wd is the best.

# *** Running the baseline ***
# learning_rate=2e-5
# python -m torch.distributed.launch --nproc_per_node 4 run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/simplification_results/baseline_lr_"${learning_rate}"_no_wd" --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 2 


# *** Running the baseline for base size ***
learning_rate=2e-5
max_steps=300000
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path facebook/bart-base --do_train --do_eval --do_predict --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/simplification_results/baseline_base_lr_"${learning_rate}"_no_wd_max_steps_"${max_steps} --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 20000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 20000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 1 

max_steps=200000
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path facebook/bart-base --do_train --do_eval --do_predict --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/simplification_results/baseline_base_lr_"${learning_rate}"_no_wd_max_steps_"${max_steps} --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 20000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 20000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 1

# above ones was wrong.
learning_rate=2e-5
max_steps=300000
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path facebook/bart-base --do_train --do_eval --do_predict --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/simplification_results/corrected_baseline_base_lr_"${learning_rate}"_no_wd_max_steps_"${max_steps} --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 20000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 20000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 1 

max_steps=200000
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path facebook/bart-base --do_train --do_eval --do_predict --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/simplification_results/corrected_baseline_base_lr_"${learning_rate}"_no_wd_max_steps_"${max_steps} --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 20000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 20000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 1




# evaluate the baseline model.
'''
for TOP_P in 0.9 0.95 0.99 
do 
	for TEMPERATURE in 1 2 4 
	do
        learning_rate=2e-5
	max_steps=60000
	model_name="${LOCAL_DIR}/simplex-diffusion/simplification_results/baseline_2e-5/checkpoint-400000"
        python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_name} --do_predict --do_eval --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/simplification_results/baseline_lr_"${learning_rate}"_no_wd" --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 2  --top_p ${TOP_P} --temperature ${TEMPERATURE} 
	done
done
'''
# For not having top-p, we should not pass it.
#for TEMPERATURE in 1 2 4 
#do
learning_rate=2e-5
max_steps=60000
model_name="${LOCAL_DIR}/simplex-diffusion/simplification_results/baseline_2e-5/checkpoint-400000"
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_name} --do_predict --do_eval --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/simplification_results/baseline_lr_"${learning_rate}"_no_wd" --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --max_eval_samples 96 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 2  #--temperature ${TEMPERATURE} 
#done


# Running the baseline on wiki-alignment
learning_rate=3e-5
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path facebook/bart-base --do_train --do_eval --do_predict --dataset_name wiki_alignment  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/wiki_alignment_tune_lr/roberta_base_lr_"${learning_rate}"_no_wd" --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 80000 --max_eval_samples 96 --max_source_length 128  --max_target_length 128  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 1  --dataset_folder "${LOCAL_DIR}/simplex-diffusion/datasets/wiki_alignment/"


# ****** this is reported for wiki-alignment ********
# evaluate the model.
model_path="${LOCAL_DIR}/outputs/paper_experiments/wiki_alignment_tune_lr/roberta_base_lr_3e-5_no_wd/checkpoint-80000"
learning_rate=3e-5
#python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path ${model_path} --do_eval --do_predict --dataset_name wiki_alignment  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/wiki_alignment_tune_lr/roberta_base_lr_"${learning_rate}"_no_wd" --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 80000 --max_eval_samples 96 --max_source_length 128  --max_target_length 128  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 1  --dataset_folder "${LOCAL_DIR}/simplex-diffusion/datasets/wiki_alignment/"

#====================================================================
# running the baseline bart-base for the dataset qqp.
learning_rate=3e-5
max_steps=90000
# python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path facebook/bart-base --do_train --do_eval --do_predict --dataset_name qqp  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/qqp_tune_steps/bart_base_lr_"${learning_rate}"_no_wd_max_steps_"${max_steps} --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 100  --max_target_length 85  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 1  --dataset_folder "${LOCAL_DIR}/simplex-diffusion/datasets/qqp/"


# running the baseline on qg.
learning_rate=3e-5
max_steps=120000 # 120000
# python -m torch.distributed.launch --nproc_per_node 8 run_simplification.py --model_name_or_path facebook/bart-base --do_train --do_eval --do_predict --dataset_name qg  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/qg_tune_steps/bart_base_lr_"${learning_rate}"_no_wd_max_steps_"${max_steps} --per_device_train_batch_size=1 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 155  --max_target_length 65  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 --predict_with_generate  ${PARAMS_FOR_LOCAL} --save_checkpoints_on_s3 --gradient_accumulation_steps 1  --dataset_folder "${LOCAL_DIR}/simplex-diffusion/datasets/qg/"





#====================================================================

# Run summarization.
# data length=512
learning_rate=2e-5
max_steps=60000
# max position embedding for BART is 1024, so we do not need to resize it.
#python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir "${LOCAL_DIR}/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps} --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3 


# run roberta-base for more iterations.
# max position embedding for BART is 1024, so we do not need to resize it.
learning_rate=3e-5
max_steps=170000
model_name=facebook/bart-base
#python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${model_name} --do_train --do_eval --do_predict --dataset_name xsum --dataset_config "3.0.0" --output_dir "${LOCAL_DIR}/outputs/paper_experiments/new_summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps}"_model_bart_base" --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 10000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 10000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3 



# run for smaller model.
learning_rate=2e-5
max_steps=60000
# max position embedding for BART is 1024, so we do not need to resize it.
model_name=facebook/bart-base
#python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${model_name} --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir "${LOCAL_DIR}/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps}"_model_"${model_name} --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3 



# run summarization on cnn-dailymail
# data length=512
learning_rate=3e-5
max_steps=120000
model_name=facebook/bart-base
#python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${model_name} --do_train --do_eval --do_predict --dataset_name "cnn_dailymail" --dataset_config "3.0.0" --output_dir "${LOCAL_DIR}/outputs/paper_experiments/cnn_dailymail_results/baseline_bart_base_lr_"${learning_rate}"_max_steps_"${max_steps} --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 20000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 20000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3 

model_path="${LOCAL_DIR}/outputs/paper_experiments/cnn_dailymail_results/baseline_bart_base_lr_3e-5_max_steps_120000/checkpoint-120000/"
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${model_path} --do_eval --do_predict --dataset_name "cnn_dailymail" --dataset_config "3.0.0" --output_dir ${model_path}"/generations/" --per_device_train_batch_size=6 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 20000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 20000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3 
 


# evaluate the base model.
: '
for TOP_P in 0.9 0.95 0.99
do 
	for TEMPERATURE in 1 2 4 
	do
        learning_rate=2e-5
	max_steps=60000
	model_name="${LOCAL_DIR}/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps}"_model_facebook/bart-base/checkpoint-60000/"
        python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${model_name}  --do_eval --do_predict --dataset_name xsum --dataset_config "3.0.0" --output_dir "${LOCAL_DIR}/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps}"_model_facebook/bart-base" --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3 --top_p ${TOP_P} --temperature ${TEMPERATURE}
	done
done

# evaluating for top-p=None
	for TEMPERATURE in 1 2 4 
	do
        learning_rate=2e-5
	max_steps=60000
	model_name="${LOCAL_DIR}/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps}"_model_facebook/bart-base/checkpoint-60000/"
        python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${model_name}  --do_eval --do_predict --dataset_name xsum --dataset_config "3.0.0" --output_dir "${LOCAL_DIR}/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps}"_model_facebook/bart-base" --per_device_train_batch_size=6 --per_device_eval_batch_size=12 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3  --temperature ${TEMPERATURE}
	done
'

: '
# Evaluate the summarizaiton.
learning_rate=2e-5
max_steps=60000
# max position embedding for BART is 1024, so we do not need to resize it.
for learning_rate in 2e-5 
do
   for checkpoint_id in 50000 55000 60000
   do 	   
   checkpoint="${LOCAL_DIR}/outputs/paper_experiments/summarization_results/baseline_lr_"${learning_rate}"_steps_"${max_steps}"/checkpoint-"${checkpoint_id}
   python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py --model_name_or_path ${checkpoint} --do_predict --dataset_name xsum --dataset_config "3.0.0" --output_dir ${checkpoint} --per_device_train_batch_size=6 --per_device_eval_batch_size=12   --report_to tensorboard --eval_steps 1000  --max_steps ${max_steps} --max_eval_samples 96 --max_source_length 392  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --weight_decay 0.0 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}  --gradient_accumulation_steps 1  --predict_with_generate --save_checkpoints_on_s3 --load_states_in_eval_from_model_path true 
   done
done
'

#####################################################################
# *** Running glue baseline ****
#DATASET="sst2"
#python run_glue.py --model_name_or_path roberta-large  --dataset_name ${DATASET} --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy epoch --save_strategy epoch  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/glue_results/baseline_"${DATASET} --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 3e-5 --num_train_epochs 3 --logging_steps 50  --load_best_model_at_end true --checkpoint_best_model --greater_is_better true --warmup_steps 500  --tokenizer_name roberta-large --save_total_limit 1 --lr_scheduler_type linear  --gradient_accumulation_steps 2

# 10 epochs for mrpc and rte, wnli,stsb cola, 3 for the rest.
# DATASET="stsb"
#python run_glue.py --model_name_or_path roberta-large  --dataset_name ${DATASET} --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy epoch --save_strategy epoch  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/glue_results/baseline_"${DATASET} --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 3e-5 --num_train_epochs 10 --logging_steps 50  --load_best_model_at_end true --checkpoint_best_model --greater_is_better true --warmup_steps 500  --tokenizer_name roberta-large --save_total_limit 1 --lr_scheduler_type linear  --gradient_accumulation_steps 2

#####################################################################

# DEBUG
# python run_simplification.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name wikilarge  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/debug/" --per_device_train_batch_size=12 --per_device_eval_batch_size=25 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 500000 --num_train_epochs 5 --max_source_length 64  --max_target_length 64  --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate ${learning_rate} --pad_to_max_length  --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL}   --predict_with_generate  --weight_decay 0.01 --eval_steps 10 

# Debug
#python run_summarization.py --model_name_or_path facebook/bart-large --do_train --do_eval --dataset_name xsum --dataset_config "3.0.0" --output_dir debug  --per_device_train_batch_size=6 --per_device_eval_batch_size=15 --overwrite_output_dir  --report_to tensorboard --eval_steps 1000  --max_steps 1000000 --max_eval_samples 96 --max_source_length 512  --max_target_length 120   --evaluation_strategy steps  --lr_scheduler_type linear --learning_rate 1e-4 --pad_to_max_length  --weight_decay 0.01 --warmup_steps 2000 --logging_steps 50 --save_steps 1000 ${PARAMS_FOR_LOCAL} --resize_position_embeddings True --fp16 --gradient_accumulation_steps 2  --predict_with_generate --eval_steps 10

# DEBUG
# DATASET="mnli"
# python run_glue.py --model_name_or_path roberta-large  --dataset_name ${DATASET} --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --evaluation_strategy epoch --save_strategy epoch  --output_dir "${LOCAL_DIR}/outputs/paper_experiments/debug" --report_to tensorboard  --overwrite_output_dir --pad_to_max_length --learning_rate 3e-5 --num_train_epochs 3 --logging_steps 50  --load_best_model_at_end true --checkpoint_best_model --greater_is_better true --warmup_steps 500  --tokenizer_name roberta-large --save_total_limit 1 --lr_scheduler_type linear  --gradient_accumulation_steps 2
