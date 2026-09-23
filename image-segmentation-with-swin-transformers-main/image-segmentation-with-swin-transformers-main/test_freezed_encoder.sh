yes | python config_changer.py --epochs=150  --model_name="SwinTUNet-ss-fe-pretrain-imagenet" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="SwinTUNet-vq-ss-fe-pretrain-imagenet" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="SwinTUNet-ss-fe-pretrain-EsViT" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="SwinTUNet-vq-ss-fe-pretrain-EsViT" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="MASWinTNet-ss-fe-pretrain-imagenet" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="MASWinTNet-vq-ss-fe-pretrain-imagenet" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="MASWinTNet-ss-fe-pretrain-EsViT" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="MASWinTNet-vq-ss-fe-pretrain-EsViT" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
