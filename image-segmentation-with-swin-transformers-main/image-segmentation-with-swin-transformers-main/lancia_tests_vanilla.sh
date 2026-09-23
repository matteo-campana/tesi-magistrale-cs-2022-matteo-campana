yes | python config_changer.py --epochs=150  --model_name='maresunet18-ss' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name='maresunet18-ss-vq' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name='maresunet50-ss' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name='maresunet50-ss-vq' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name='MASWinTNet-ss' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name='MASWinTNet-vq-ss' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name='MASWinTNet-nano-ss' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
yes | python config_changer.py --epochs=150  --model_name='MASWinTNet-nano-vq-ss' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=True
cat config/train_pl_ss.yaml
python main_pl_ss.py
