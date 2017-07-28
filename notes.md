# Repo Structure

## Root
 - custom scripts to run inference and training
 - utils.py (probably better to move them to separate folder)

## object_detection folder

Partially ported from tensorflow/models/object_detection

I decided to leave the filenames and structure unchanged to ensure compatibility with TF Object detection model zoo, the only difference is that the pre-trained models should NOT be extracted to this folder

## models folder

Folder to keep our "model zoo". These folder should have one .json file for each model with the following information:
* Model name
* Path to model folder
* Path to model file
* path to label map
* number of classes


## custom data folder

Folder to keep new datasets to fine-tune models to solve different problems

## Instructions

### Run
python object_detection_app.py -m './models/ssd_v1_flickr47.json'


### Train

cd/Documents/insight/project/
source ativate py27_insight

#### LOCAL VERSION
export PYTHONPATH=$PYTHONPATH:/home/bruno/Documents/tensorflow/models:/home/bruno/Documents/tensorflow/models/slim

#### PAPERSPACE VERSION
export PYTHONPATH=$PYTHONPATH:/home/paperspace/Documents/models:/home/paperspace/Documents/models/slim

##### Create training records
COMING SOON

##### Training script
python object_detection/train.py \
    --logstoderr \
    --pipeline_config_path=./ssd_mobilenet_v1_flickr47.config \
    --train_dir=training_models/ssd_mobilenet_v1_flickr47/

##### Evaluating script
python object_detection/eval.py \
    --logstoderr \
    --pipeline_config_path=./ssd_mobilenet_v1_flickr47.config \
    --eval_dir=training_models/ssd_mobilenet_v1_flickr47/ \
    --checkpoint_dir=training_models/ssd_mobilenet_v1_flickr47/

##### Tensorboard
tensorboard --logdir=./training_models/ssd_mobilenet_v1_flickr47

##### Export checkpoint to graph
python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=./ssd_mobilenet_v1_flickr47.config \
    --checkpoint_path=training_models/ssd_mobilenet_v1_flickr47/model.ckpt-17353 \
    --inference_graph_path=models/ssd_mobilenet_v1_flickr47/frozen_inference_graph.pb
