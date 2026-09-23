yes | python config_changer.py --epochs=150  --model_name='maresunet50-ss-pretrain-dino-fair' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name='maresunet50-ss-vq-pretrain-dino-fair' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer_dino.py \
--model_name_features_extractor='maresunet50-pretrain-dino-fair' --model_name_semantic_segmentation='maresunet50-ss-pretrain-dino-fair' \
--train_dataset_length=-1 --valid_dataset_length=-1 \
--vq_flag=False
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
yes | python config_changer_dino.py \
--epochs=150 \
--model_name_features_extractor='maresunet50-vq-pretrain-dino-fair' --model_name_semantic_segmentation='maresunet50-ss-vq-pretrain-dino-fair' \
--train_dataset_length=-1 --valid_dataset_length=-1 \
--vq_flag=True
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
