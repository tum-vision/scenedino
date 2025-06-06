#!/bin/bash

echo "----------------------- Downloading pretrained model -----------------------"

model=$1

if [[ $model == "ssc-kitti-360" ]]
then
  cp_link="https://cvg.cit.tum.de/webshare/u/jevtic/scenedino/checkpoints/seg-best-dino/checkpoint.pt"
  cfg_link="https://cvg.cit.tum.de/webshare/u/jevtic/scenedino/checkpoints/seg-best-dino/training_config.yaml"

  cp_download_path="out/scenedino-pretrained/seg-best-dino/checkpoint.pt"
  cfg_download_path="out/scenedino-pretrained/seg-best-dino/training_config.yaml"

elif [[ $model == "scenedino-kitti-360" ]]
then
  cp_link="https://cvg.cit.tum.de/webshare/u/jevtic/scenedino/checkpoints/feat-dino/checkpoint.pt"
  cfg_link="https://cvg.cit.tum.de/webshare/u/jevtic/scenedino/checkpoints/feat-dino/training_config.yaml"

  cp_download_path="out/scenedino-pretrained/feat-dino/checkpoint.pt"
  cfg_download_path="out/scenedino-pretrained/feat-dino/training_config.yaml"

elif [[ $model == "scenedinov2-kitti-360" ]]
then
  cp_link="https://cvg.cit.tum.de/webshare/u/jevtic/scenedino/checkpoints/feat-dinov2/checkpoint.pt"
  cfg_link="https://cvg.cit.tum.de/webshare/u/jevtic/scenedino/checkpoints/feat-dinov2/training_config.yaml"

  cp_download_path="out/scenedino-pretrained/feat-dinov2/checkpoint.pt"
  cfg_download_path="out/scenedino-pretrained/feat-dinov2/training_config.yaml"

else
  echo Unknown model: $model
  echo Possible options: \"ssc-kitti-360\", \"scenedino-kitti-360\", \"scenedinov2-kitti-360\"
  exit
fi

basedir=$(dirname $0)
cp_outdir=$(dirname $cp_download_path)
cfg_outdir=$(dirname $cfg_download_path)

cd $basedir || exit
echo Operating in \"$(pwd)\".
echo Creating directories.

mkdir -p $cp_outdir
echo Downloading checkpoint from \"$cp_link\" to \"$cp_download_path\".
wget -O $cp_download_path $cp_link

mkdir -p $cfg_outdir
echo Downloading config from \"$cfg_link\" to \"$cfg_download_path\".
wget -O $cfg_download_path $cfg_link
