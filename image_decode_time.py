import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import ops
import tensorflow.contrib.stat_summarizer

def load(file_name):
  with ops.Graph().as_default() as graph:

    file_reader = tf.read_file(file_name, 'file_reader')

    if file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    elif file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, name='png_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')

    graph_def = graph.as_graph_def()
    stat = pywrap_tensorflow.NewStatSummarizer(graph_def.SerializeToString())

    with tf.Session() as sess:
        for i in range(0, 100):
            run_metadata = tf.RunMetadata()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            result = sess.run(image_reader, {}, run_options, run_metadata)
            step_stats = run_metadata.step_stats
            stat.ProcessStepStatsStr(step_stats.SerializeToString())

    stat.PrintStepStats()

if __name__ == "__main__":
    load(sys.argv[1])
