"""
Demo script that starts a server which exposes liver segmentation.

Based off of https://github.com/morpheus-med/vision/blob/master/ml/experimental/research/prod/model_gateway/ucsd_server.py
"""

import functools
import logging
import logging.config
import os
import tempfile
import yaml
import json
import numpy
import pydicom

from utils.image_conversion import convert_to_nifti

from utils import tagged_logger
import tensorflow as tf
# ensure logging is configured before flask is initialized
print(tf.__version__)

with open('logging.yaml', 'r') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)

logger = logging.getLogger('inference')

# pylint: disable=import-error,no-name-in-module
from gateway import Gateway
from flask import make_response
import cv2
import numpy as np

def handle_exception(e):
    logger.exception('internal server error %s', e)
    return 'internal server error', 500

def get_empty_response():
    response_json = {
        'protocol_version': '1.0',
        'parts': []
    }
    return response_json, []

def get_bounding_box_2d_response(json_input, dicom_instances):
    base_model = tf.keras.models.load_model('./coviddetector/models_dicom/covid19_dcm_test_9_9_986_988.h5')
    height = 224
    width = 224
    dim = np.zeros((height, width))
    res=[]
    response_json = {
        'protocol_version': '1.0',
        'parts': [],
        'bounding_boxes_2d': []
    }
    for instances in dicom_instances:
        dcm = pydicom.read_file(instances)
        dataset = dcm.pixel_array
        img = cv2.resize(dataset, (height, width))

        prediction = base_model.predict(np.array(np.reshape(img,(1, img.shape[0], img.shape[1], 1))))
        if np.argmax(prediction, axis=1)==0:
            label='negative'
        elif np.argmax(prediction, axis=1)==1:
            label='positive'

        response_json['bounding_boxes_2d'].append(
            {
                'SOPInstanceUID': dcm.SOPInstanceUID,
                'top_left': [0, 0],
                'bottom_right': [dataset.shape[0], dataset.shape[1]],
                'label': label
            }
        )


    return response_json, []


def request_handler(json_input, dicom_instances, input_digest):
    """
    A mock inference model that returns a mask array of ones of size (height * depth, width)
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({ 'input_hash': input_digest })
    transaction_logger.info('mock_model received json_input={}'.format(json_input))

    # If your model accepts Nifti files as input then uncomment the following lines:
    # convert_to_nifti(dicom_instances, 'nifti_output.nii')
    # print("Converted file to nifti 'nifti_output.nii'")
    
    if json_input['inference_command'] == 'get-bounding-box-2d':
        return get_bounding_box_2d_response(json_input, dicom_instances)
    else:
        return get_empty_response()


if __name__ == '__main__':
    app = Gateway(__name__)
    app.register_error_handler(Exception, handle_exception)
    app.add_inference_route('/', request_handler)

    app.run(host='0.0.0.0', port=8002   , debug=True, use_reloader=True)
