# Davis: Deep Learning Analytics for Video Streams

### Recognizing brands in live video streams

## Motivation

According to Cisco, internet video was 51% of the web traffic in 2016, and they expect this number to grow to 67% by 2021 [[1]](https://www.recode.net/2017/6/8/15757594/future-internet-traffic-watch-live-video-facebook-google-netflix). With so much content being delivered in this media format, there will be a clear need for software that can process videos and produce meaningful reports.

Davis addresses the very first step in this pipeline, recognizing brands in live video streams to help companies understand how much visibility their products are getting.

## Building Davis

### Architecture

Davis was built on top of Google's open source object recognition API [[2]](https://github.com/tensorflow/models/tree/master/object_detection). I had to make some changes to their original distribution to improve the training performance, but these changes are minor and if you are already familiar with their API you can use the official version.


### Dataset

I leveraged Google's pre-trained models on MS COCO and fine tuned a few different models on the FlickrLogos-47 dataset [[3]](http://www.multimedia-computing.de/flickrlogos/).


### Model Weights

I am providing weights for a model optimized for real-time detection. This model can process up to 20 fps in a GTX1080 and achieves a 0.77 mAP on the Flickr logos dataset.

Weights for different models are coming soon.

### Serving

These models can be served in two ways, included in this repo is a local-only app that uses OpenCV to access your webcam and stream annotated frames back to the user.

There is a web-app on this [repo](https://github.com/bguisard/davis_app) that serves the model using Flask and WebRTC instead.

Building a GUI-enabled instance of OpenCV with all the correct bindings can be a hassle, so using the web-app may be a good option even if you intend to just play around with the model locally.

The following links were extremely helpful in creating these two versions [[4]](https://medium.com/towards-data-science/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32), [[5]](https://blog.miguelgrinberg.com/post/video-streaming-with-flask), [[6]](https://blog.miguelgrinberg.com/post/flask-video-streaming-revisited).

### Compatibility

Davis with compatibility in mind and it should work seamlessly with any TensorFlow frozen model and any image input size.


## Dependencies

On top of the dependencies listed on requirements.txt you will need OpenCV 3 with GUI enabled if you want to use the local version of the app.

```
pip install -r requirements.txt
```

## Instructions

The steps below will give a brief description of how to use the model. Please refer to Google's object detection API for a more detailed walk-through of how to train it on your own dataset.

### 1 - Convert your dataset to TFRecord
```
python create_flickrlogos_tf_record \
    --label_map_path=PATH_TO_DATASET_LABELS \
    --data_dir=PATH_TO_DATA_FOLDER \
    --output_path=PATH_TO_OUTPUT_FILE
```

### 2 - Train the model
```
python object_detection/train.py \
    --logstoderr \
    --pipeline_config_path=PATH_TO_MODEL.CONFIG \
    --train_dir=PATH_TO_MODEL
```

### 3 - "Publishing" the model

```
python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=PATH_TO_MODEL.CONFIG \
    --checkpoint_path=PATH_TO_MODEL_CHECKPOINT \
    --inference_graph_path=PATH_TO_PUBLISH
```

### 4 - Launching the App
```
python object_detection_app.py -m 'PATH_TO_MODEL_DESCRPITION.json'
```
