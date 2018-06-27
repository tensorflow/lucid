import tensorflow as tf


def create_session(target='', timeout_sec=10):
  '''Create an intractive TensorFlow session.
  
  Helper function that creates TF session that uses growing GPU memory
  allocation and opration timeout. 'allow_growth' flag prevents TF
  from allocating the whole GPU memory an once, which is useful
  when having multiple python sessions sharing the same GPU.
  '''
  graph = tf.Graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.operation_timeout_in_ms = int(timeout_sec*1000)
  return tf.InteractiveSession(target=target, graph=graph, config=config)
  
