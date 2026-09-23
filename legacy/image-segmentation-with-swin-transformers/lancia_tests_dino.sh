yes | python config_changer_dino.py \
 --epochs=150 \
 --model_name_features_extractor='maresunet18' --model_name_semantic_segmentation='maresunet18-ss' \
 --train_dataset_length=-1 --valid_dataset_length=-1 \
 --vq_flag=False
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
yes | python config_changer_dino.py \
 --epochs=150 \
 --model_name_features_extractor='maresunet18-vq' --model_name_semantic_segmentation='maresunet18-ss-vq' \
 --train_dataset_length=-1 --valid_dataset_length=-1 \
 --vq_flag=True
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
yes | python config_changer_dino.py \
 --epochs=150 \
 --model_name_features_extractor='maresunet50' --model_name_semantic_segmentation='maresunet50-ss' \
 --train_dataset_length=-1 --valid_dataset_length=-1 \
 --vq_flag=False
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
yes | python config_changer_dino.py \
 --epochs=150 \
 --model_name_features_extractor='maresunet50-vq' --model_name_semantic_segmentation='maresunet50-ss-vq' \
 --train_dataset_length=-1 --valid_dataset_length=-1 \
 --vq_flag=True
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
--model_name_features_extractor='MASWinTNet' --model_name_semantic_segmentation='MASWinTNet-ss' \
--train_dataset_length=-1 --valid_dataset_length=-1 \
--vq_flag=False
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
yes | python config_changer_dino.py \
--epochs=150 \
--model_name_features_extractor='MASWinTNet-vq' --model_name_semantic_segmentation='MASWinTNet-vq-ss' \
--train_dataset_length=-1 --valid_dataset_length=-1 \
--vq_flag=True
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
--model_name_features_extractor='MASWinTNet-nano' --model_name_semantic_segmentation='MASWinTNet-nano-ss' \
--train_dataset_length=-1 --valid_dataset_length=-1 \
--vq_flag=False
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
yes | python config_changer_dino.py \
--epochs=150 \
--model_name_features_extractor='MASWinTNet-nano-vq' --model_name_semantic_segmentation='MASWinTNet-nano-vq-ss' \
--train_dataset_length=-1 --valid_dataset_length=-1 \
--vq_flag=True
cat config/train_pl_features_extractor.yaml
python main_pl_ss_dino.py
