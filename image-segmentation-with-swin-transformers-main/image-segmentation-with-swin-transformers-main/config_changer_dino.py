import argparse
import yaml
import os
from pathlib import Path

def main():

    assert  Path("config/train_pl_features_extractor.yaml").is_file() , "config file not found"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_features_extractor",default="maresunet18")
    parser.add_argument("--model_name_semantic_segmentation",default="maresunet18-ss")
    parser.add_argument("--epochs",default=100,type=int)
    parser.add_argument("--train_dataset_length",default=-1,type=int)
    parser.add_argument("--valid_dataset_length",default=-1,type=int)
    parser.add_argument("--vq_flag",default=False)

    args = parser.parse_args()
    print(args)

    model_name_features_extractor = args.model_name_features_extractor
    model_name_semantic_segmentation = args.model_name_semantic_segmentation
    epochs = int(args.epochs)
    train_dataset_length = int(args.train_dataset_length)
    valid_dataset_length = int(args.valid_dataset_length)
    vq_flag = args.vq_flag

    if vq_flag == "True":
        vq_flag = True
    elif vq_flag == "False":
        vq_flag = False

    string = str(input())
    with open("config/train_pl_features_extractor.yaml") as f:
        yaml_data = yaml.safe_load(f)

        if yaml_data["model_features_extractor"] != model_name_features_extractor:
             yaml_data["model_features_extractor"] = model_name_features_extractor
        if yaml_data["model_semantic_segmentation"] != model_name_semantic_segmentation:
             yaml_data["model_semantic_segmentation"] = model_name_semantic_segmentation
        if yaml_data["epochs"] != epochs:
            yaml_data["epochs"] = epochs
        if yaml_data["train_dataset_length"] != train_dataset_length:
            yaml_data["train_dataset_length"] = train_dataset_length
        if yaml_data["valid_dataset_length"] != valid_dataset_length:
            yaml_data["valid_dataset_length"] = valid_dataset_length
        if yaml_data["vq_flag"] != vq_flag:
            yaml_data["vq_flag"] = vq_flag

        f.close()

    with open("config/train_pl_features_extractor.yaml", "w") as f:
        yaml.dump(yaml_data, f)
        f.close()

if __name__ == "__main__":
    main()
