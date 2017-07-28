# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Flickr Logos dataset to TFRecord for object_detection.

See: Stefan Romberg, Lluis Garcia Pueyo, Rainer Lienhart, Roelof van Zwol
     Scalable Logo Recognition in Real-World Images
     http://www.multimedia-computing.de/flickrlogos/

Example usage:
    ./python create_flickrlogos_tf_record \
        --label_map_path=PATH_TO_DATASET_LABELS \
        --data_dir=PATH_TO_DATA_FOLDER \
        --output_path=PATH_TO_OUTPUT_FILE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw Flickr dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set or merged set.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'models/labels/flickrlogos47_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'test']


def parse_annotations(image_path):
    annotation_path = image_path[:-3]+'gt_data.txt'

    with tf.gfile.GFile(annotation_path) as fid:
        lines = fid.readlines()

    annos = [line.strip().split(' ') for line in lines]

    annotations = []

    for anno in annos:
        a = {}
        a['xmin'] = int(anno[0])
        a['ymin'] = int(anno[1])
        a['xmax'] = int(anno[2])
        a['ymax'] = int(anno[3])
        a['class'] = str(anno[4])
        a['mask_number'] = str(anno[6])
        a['difficult'] = int(anno[7])
        a['truncated'] = int(anno[8])
        a['pose'] = 'Frontal'  # hard-coded to conform with API
        annotations.append(a)

    return annotations


def get_label_map_text(label_map_path):
    """Reads a label map and returns a dictionary of label names to display name.

    Args:
      label_map_path: path to label_map.

    Returns:
      A dictionary mapping label names to id.
    """
    label_map = label_map_util.load_labelmap(label_map_path)
    label_map_text = {}
    for item in label_map.item:
        label_map_text[item.name] = item.display_name
    return label_map_text


def create_label_map(labels_file, output_name='flickrlogos_label_map.pbtxt'):
    """ Creates a proto label map based on the labels file from
        the Flickr dataset.

        name: is the code that the class is referred in the dataset
        id: is the id we will use inside our TF model (starts at 1)
        display_name: name to be displayed
    """

    with tf.gfile.GFile(labels_file) as fid:
        lines = fid.readlines()

    classes = [line.strip().split('\t') for line in lines]

    with open(output_name, 'wb') as text_file:
        for cls in classes:
            text_file.write("item {\n")
            text_file.write("  name: {}\n".format("\""+cls[1]+"\""))
            class_id = str(int(cls[1]) + 1)
            text_file.write("  id: {}\n".format(class_id))
            text_file.write("  display_name: {}\n".format("\""+cls[0]+"\""))
            text_file.write("}\n")


def data_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       label_map_text,
                       ignore_difficult_instances=False):
    """Convert TXT derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: TO REPLACE - dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = data['img_path']

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_png = fid.read()

    encoded_png_io = io.BytesIO(encoded_png)

    image = PIL.Image.open(encoded_png_io)

    if image.format != 'PNG':
        raise ValueError('Image format not PNG')

    key = hashlib.sha256(encoded_png).hexdigest()

    width = image.width
    height = image.height

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    for obj in data['annos']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))
        xmin.append(float(obj['xmin']) / width)
        ymin.append(float(obj['ymin']) / height)
        xmax.append(float(obj['xmax']) / width)
        ymax.append(float(obj['ymax']) / height)
        classes_text.append(label_map_text[obj['class']].encode('utf8'))
        classes.append(label_map_dict[obj['class']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['img_path'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['img_path'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    label_map_text = get_label_map_text(FLAGS.label_map_path)

    logging.info('Reading from FlickLogos dataset.')
    examples_path = os.path.join(data_dir, FLAGS.set, 'filelist.txt')
    set_dir = os.path.join(data_dir, FLAGS.set)
    examples_list = dataset_util.read_examples_list(examples_path)

    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        img_path = os.path.join(set_dir, example[2:])
        anno_path = img_path[:-3] + 'txt'

        data = {}
        data['img_path'] = img_path
        data['annos'] = parse_annotations(img_path)

        tf_example = data_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                        label_map_text,
                                        FLAGS.ignore_difficult_instances)

        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
