# AI Insight Fellowship - Summer 2017

## DeepBrand

For information about the repo structure and using instructions check [notes.md](./notes.md)

## Project steps

### Brand Recognition in Video Stream

#### Idea
Train an end-to-end pipeline that is able to effectively and efficiently locate brand(s) in a video stream.

##### Breaking the problem

###### I) Identify Brands in Live video
* Minimum Performance Requirement: 30+ fps

* Models - YOLO, SSD, Faster RCNN

* Leverage from pre-trained models to speed training

###### II) scalability
* 300 hours of video are uploaded to YouTube every minute

* $44 million in GPUs needed to process this in real-time (30fps)

* NoScope can increase processing speeds up to 1,000x for specific cases

###### III) Semantic Analysis
* Scan video sections where the brands were identified and detect possible cases of brand abuse

* Send alert to brand manager if Pabuse > threshold

#### Deliverables

###### I) Identify Brands in Live video
* Website that can process live video stream and identify selected brand(s)

###### II) scalability
* e-mail / slack bot post with all youtube videos posted on a given day with the link to the video and the time that the brand(s) appeared

###### III) Semantic Analysis
* Improve from step II, only sending alerts for abusive videos

#### Datasets

* MS COCO
* FlickrLogos-47

#### TODO:
* Change name, there are other 2 companies with this name already
* Clean notes.md

#### Useful Resources

- Inference Optimization: fusion, quantization, reduced precision

#### References
[1] [Tensorflow Models](https://github.com/tensorflow/models)

[2] [Building a Real-Time Object Recognition App with Tensorflow and OpenCV](https://medium.com/towards-data-science/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32)

[3] [Increasing webcam FPS with Python and OpenCV](http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)

[4] [WebRTC](https://webrtc.org/)

[5] [Speed/accuracy trade-offs for modern convolutional object detectors](http://arxiv.org/abs/1611.10012)
