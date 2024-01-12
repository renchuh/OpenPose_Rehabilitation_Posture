import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

import imageio

from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
import time

cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
tic = time.time()
# Read image from file
file_name = "demo/image.png"
image = imageio.imread(file_name )

image_batch = data_to_input(image)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
toc = time.time()
print('processing time is %.5f' % (toc - tic))
# Visualise
visualize.show_heatmaps(cfg, image, scmap, pose)
visualize.waitforbuttonpress()
