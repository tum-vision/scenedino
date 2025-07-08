import os
import sys
import argparse
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_scenedino_checkpoint(model_name):
    print("----------------------- Downloading pretrained model -----------------------")
    model_configs = {
        "ssc-kitti-360-dino": {
            "model-dir": "seg-best-dino"
        },
        "ssc-kitti-360-dino-orb-slam": {
            "model-dir": "seg-best-dino-orb-slam"
        },
        "ssc-kitti-360-dinov2": {
            "model-dir": "seg-best-dinov2"
        }
    }

    repo_id = "jev-aleks/SceneDINO"
    checkpoint_filename = "checkpoint.pt"
    config_filename = "training_config.yaml"
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Possible options: {', '.join(model_configs.keys())}")
    
    config = model_configs[model_name]
    
    output_dir = Path("out/scenedino-pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = Path(config["model-dir"]) / checkpoint_filename
    config_filename = Path(config["model-dir"]) / config_filename

    checkpoint_path = output_dir / checkpoint_filename
    config_path = output_dir / config_filename
    
    print(f"Operating in \"{os.getcwd()}\".")
    print(f"Creating directories: {output_dir}")
    
    # Download checkpoint
    print(f"Downloading checkpoint from HF repo \"{repo_id}\" to \"{checkpoint_path}\".")
    hf_hub_download(
        repo_id=repo_id,
        filename=str(checkpoint_filename),
        local_dir=str(output_dir),
    )
    
    # Download config
    print(f"Downloading config from HF repo \"{repo_id}\" to \"{config_path}\".")
    hf_hub_download(
        repo_id=repo_id, 
        filename=str(config_filename),
        local_dir=str(output_dir),
    )
    
    print("Download completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained models from Hugging Face Hub")
    parser.add_argument("model", help="Model name to download")
    
    args = parser.parse_args()
    download_scenedino_checkpoint(args.model)


if __name__ == "__main__":
    main()