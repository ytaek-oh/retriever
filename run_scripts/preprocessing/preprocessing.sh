DATA_ROOT=$1

# after downloading official nice dataset on ${DATA_PATH}/nice, create json annotations
python utils/create_nice_annotations.py --data_path ${DATA_PATH}/nice

# Then download the 1M of shutterstock dataset from web.
# follow the instruction from the jupyter notebook file located in 
#    'nice/datasets/download_scripts/DownloadShutterstockCaptions/shutterstock_preprocess.ipynb'
# to create 'shutterstock_1m.json'.

# after downloading shutterstock dataset, construct FAISS index file
CUDA_VISIBLE_DEVICES=0,1,2,3 python faiss_construct.py \
    --cfg-path nice/configs/preprocessing/faiss/blip2_construct_shutterstock_1m.yaml \
    --options run.data_root ${DATA_ROOT}

# dataset discovery and creating annotations for retrieval-based fusion, based on the constructed FAISS index.
CUDA_VISIBLE_DEVICES=0 python faiss_assign.py \
    --cfg-path nice/configs/preprocessing/faiss/blip2_assign_shutterstock_1m_to_eval_data.yaml \
    --options run.data_root ${DATA_ROOT}

CUDA_VISIBLE_DEVICES=0 python faiss_assign.py \
    --cfg-path nice/configs/preprocessing/faiss/blip2_assign_shutterstock_1m_to_discovery_data.yaml \
    --options run.data_root ${DATA_ROOT}


# extract caption features for retrieval-based fusion
# this command gives 'text_feature_nice_{val,test}.h5' on '../datasets/shutterstock/caption_features'
python -m torch.distributed.run --nproc_per_node=1 --master_port=29502 extract_feats.py \
  --cfg-path nice/configs/preprocessing/qformer_extraction/qformer_feats_for_nice_eval.yaml \
  --options run.world_size 1 \
  run.data_root ${DATA_ROOT}


# concat nice val + discovery -> for training and evaluation
# split 4k train, 1k val
python utils/concatenate_dataset_with_feature.py --data_root ${DATA_ROOT} \
    --anns_first ${DATA_ROOT}/nice/nice_val_ret_ids.json \
    --feats_first ${DATA_ROOT}/shutterstock/caption_features/text_feature_nice_val.h5 \
    --anns_second ${DATA_ROOT}/nice/discovery_nice_test_top1_ret_ids.json \
    --feats_second ${DATA_ROOT}/shutterstock/caption_features/text_feature_nice_discovery_top1.h5 \
    --anns_save_name discovery_nice_test_top1_add_valid5k_ret_ids.json \
    --feats_save_name text_feature_nice_discovery_top1_add_valid5k.h5

python utils/concatenate_dataset_with_feature.py --data_root ${DATA_ROOT} \
    --anns_first ${DATA_ROOT}/nice/nice_val_split_train_ret_ids.json \
    --anns_second ${DATA_ROOT}/nice/discovery_nice_test_top1_ret_ids.json \
    --anns_save_name discovery_nice_test_top1_add_split4k_ret_ids.json
