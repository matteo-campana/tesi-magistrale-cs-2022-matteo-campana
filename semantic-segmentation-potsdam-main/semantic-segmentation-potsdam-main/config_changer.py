import argparse
import yaml
import os
from pathlib import Path

def main():

    assert  Path("config/train_pl_ss.yaml").is_file() , "config file not found"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default="maresunet18-ss")
    parser.add_argument("--epochs",default=100,type=int)
    parser.add_argument("--train_dataset_length",default=-1,type=int)
    parser.add_argument("--valid_dataset_length",default=-1,type=int)
    parser.add_argument("--vq_flag",default=False)

    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    epochs = int(args.epochs)
    train_dataset_length = int(args.train_dataset_length)
    valid_dataset_length = int(args.valid_dataset_length)
    vq_flag = args.vq_flag

    if vq_flag == "True":
        vq_flag = True
    elif vq_flag == "False":
        vq_flag = False

    string = str(input())
    with open("config/train_pl_ss.yaml") as f:
        yaml_data = yaml.safe_load(f)
        '''
        model_name = "maresunet18-ss-vq"
        epochs = 10
        train_dataset_length = 50
        valid_dataset_length = 10
        vq_flag = True
        '''
        if yaml_data["model"] != model_name:
            yaml_data["model"] = model_name
        if yaml_data["epochs"] != epochs:
            yaml_data["epochs"] = epochs
        if yaml_data["train_dataset_length"] != train_dataset_length:
            yaml_data["train_dataset_length"] = train_dataset_length
        if yaml_data["valid_dataset_length"] != valid_dataset_length:
            yaml_data["valid_dataset_length"] = valid_dataset_length
        if yaml_data["vq_flag"] != vq_flag:
            yaml_data["vq_flag"] = vq_flag

        f.close()

    with open("config/train_pl_ss.yaml", "w") as f:
        yaml.dump(yaml_data, f)
        f.close()

if __name__ == "__main__":
    main()
