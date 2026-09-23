yes | python config_changer.py --epochs=150  --model_name="MASWinTNet-ss-pretrain-imagenet" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="MASWinTNet-vq-ss-pretrain-imagenet" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="MASWinTNet-ss-pretrain-EsViT" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="MASWinTNet-vq-ss-pretrain-EsViT" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="maresunet50-ss-pretrain-EsViT" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name="maresunet50-ss-vq-pretrain-EsViT" --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
