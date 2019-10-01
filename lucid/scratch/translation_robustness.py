import time

import numpy as np
import tensorflow as tf
import shelve
import nltk
import lucid.optvis.render as render
import lucid.modelzoo.vision_models as models
from lucid.modelzoo.wordnet import synset_from_id, imagenet_synset_ids, imagenet_synsets
import tensorflow_datasets as tfds

MODEL_STORAGE_BUCKET = 'gs://clarity-public/michael'  # do NOT include a trailing slash


class SimplifiedModel(models.Model):
  labels_path = "gs://modelzoo/labels/ImageNet_standard_with_dummy.txt"
  image_shape = [224, 224, 3]
  image_value_range = (0, 1)
  input_name = 'v0/model/input_images'
  is_BGR = False

  def __init__(self, model_path):
      self._model_path = model_path
      super(SimplifiedModel, self).__init__()

  @property
  def model_path(self):
    return self._model_path


def get_model_at_index(model_name, idx):
  paths = tf.gfile.ListDirectory(f'{MODEL_STORAGE_BUCKET}/graphs/{model_name}/')

  selected_path = f'{MODEL_STORAGE_BUCKET}/graphs/{model_name}/{paths[idx]}'
  model = SimplifiedModel(selected_path)

  print(f'selected path: {selected_path}')

  return model


def get_latest_model(model_name):
  return get_model_at_index(model_name, -1)



from collections import OrderedDict
import itertools


def model_name_with_params(pool_stride, kernel_size, pool_type, blur_conv):
  REVISION_NUM = 1  # increment when the model architecture changes significantly for these runs
  return f'rev{REVISION_NUM}-{pool_type}-{pool_stride}-{kernel_size}'


def load_all_models():
  stride_and_kernel_size_variants = [
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 5),
    (4, 6),
  ]

  pooling_options = ['l2', 'max', 'avg', 'sconv']
  # pooling_options = ['sconv']  # temp to backfill strided conv
  blur_conv_options = [False]

  all_variants = list(itertools.product(stride_and_kernel_size_variants, pooling_options, blur_conv_options))

  all_variants.append(((2, 3), 'blurconv', False))

  ret = OrderedDict()
  for i, variant in enumerate(all_variants):
    (pool_stride, kernel_size), pool_type, blur_conv = variant

    #     print(f'{i}/{len(all_variants)}: pool_stride={pool_stride}, kernel_size={kernel_size}, pool_type={pool_type}, blur_conv={blur_conv}')
    model_name = model_name_with_params(pool_stride, kernel_size, pool_type, blur_conv)
    model = get_latest_model(model_name)
    model.load_graphdef()
    ret[model_name] = (
      model,
      {
        'name': model_name,
        'pool_stride': pool_stride,
        'kernel_size': kernel_size,
        'pool_type': pool_type,
        'blur_conv': blur_conv,
      }
    )

  return ret


def make_dataset(size=(224, 224), split=None, batch=None, shuffle_files=False, random_seed=None):
  def preprocess(values):
    images = values.pop('image')
    images = tf.image.resize_images(images, size)
    images = tf.image.convert_image_dtype(images, tf.dtypes.float32) / 255.
    return dict(image=images, **values)

  imagenet_ds = tfds.load(name="imagenet2012:4.0.0",
                          split=split or tfds.Split.TRAIN,
                          data_dir="gs://openai-tfds",
                          as_dataset_kwargs={'shuffle_files': shuffle_files})
  imagenet_ds = imagenet_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if random_seed is not None:
    imagenet_ds = imagenet_ds.shuffle(128, seed=random_seed)

  dataset = imagenet_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  if batch is not None:
    dataset = dataset.batch(batch)

  dataset_iterator = dataset.make_one_shot_iterator()
  dataset_t = dataset_iterator.get_next()

  image_t = dataset_t["image"]
  filename_t = dataset_t["file_name"]
  label_t = dataset_t["label"]
  return image_t, label_t, filename_t


def imagenet_sample_set(num_images=100, batch=10, random_seed=42, use_validation_set=True, with_synsets=False):
  ret = []
  synsets = imagenet_synsets()
  synset_ids = imagenet_synset_ids()
  with tf.Graph().as_default(), tf.Session() as sess:
    split = tfds.Split.VALIDATION if use_validation_set else tfds.Split.TEST
    image_t, label_t, filename_t = make_dataset(split=split, random_seed=random_seed, batch=batch)
    while len(ret) < num_images:
      try:
        images = sess.run([image_t, label_t, filename_t])
        if with_synsets:
          [ret.append((image, label, synsets[label], synset_ids[label])) for image, label, _ in
           zip(images[0], images[1], images[2])]
        else:
          [ret.append((image, label, synset_ids[label])) for image, label, _ in zip(images[0], images[1], images[2])]
      except tf.errors.OutOfRangeError:
        raise
  return ret[:num_images]


def classify_top_n(model, input_image, n=10, undo_dummy_shift=True):
  with tf.Graph().as_default(), tf.Session() as sess:
    t_img = tf.placeholder("float32", [None, None, None, 3])
    T = render.import_model(model, t_img, t_img)
    logits = T("v0/model/output_logits").eval(feed_dict={
      t_img: input_image[None]
    })
    probabilities = sess.run(tf.nn.softmax(logits))

  top_n_indices = logits[0].argsort()[-n:][::-1]
  if undo_dummy_shift:
    top_n_indices -= 1
  top_n_probabilities = probabilities[0][top_n_indices]
  top_n_labels = [model.labels[i] for i in top_n_indices]
  return list(zip(top_n_labels, top_n_probabilities, top_n_indices))


def classify_top_n_batch(model, input_images, n=10, undo_dummy_shift=True):
  stacked_images = np.stack(input_images)
  with tf.Graph().as_default(), tf.Session() as sess:
    t_img = tf.placeholder("float32", [None, None, None, 3])
    T = render.import_model(model, t_img, t_img)
    logits = T("v0/model/output_logits").eval(feed_dict={
      t_img: stacked_images
    })
    probabilities = sess.run(tf.nn.softmax(logits))

  ret = []
  for example_idx in range(logits.shape[0]):
    top_n_indices = logits[example_idx].argsort()[-n:][::-1]
    if undo_dummy_shift:
      top_n_indices -= 1
    top_n_probabilities = probabilities[example_idx][top_n_indices]
    top_n_labels = [model.labels[i] for i in top_n_indices]
    ret.append(list(zip(top_n_labels, top_n_probabilities, top_n_indices)))
  return ret


def classification_accuracy(model, input_images_and_labels, n=10, batch_size=512):
  stacked_images = np.stack([x[0] for x in input_images_and_labels])
  expected_labels = [x[1] + 1 for x in input_images_and_labels]  # + 1 to adjust for dummy class
  with tf.Graph().as_default(), tf.Session() as sess:
    print(f'classifying {len(input_images_and_labels)} images, sess={sess}')
    t_img = tf.placeholder("float32", [None, None, None, 3])
    T = render.import_model(model, t_img, t_img)

    logits = []
    for idx in range(0, stacked_images.shape[0], batch_size):
      image_batch = stacked_images[idx:(idx + batch_size)]
      print(f'running batch of shape: {image_batch.shape}')
      batch_logits = T("v0/model/output_logits").eval(feed_dict={
        t_img: image_batch
      })
      print(f'batch_logits shape: {batch_logits.shape}')
      logits.extend(batch_logits)

    logits = np.stack(logits)
    print(f'logits shape: {logits.shape}')
    probabilities = sess.run(tf.nn.softmax(logits))

  ret = []
  for example_idx in range(logits.shape[0]):
    expected_label = expected_labels[example_idx]
    example_probability = probabilities[example_idx][expected_label]

    example_rank = np.where(logits[example_idx].argsort()[::-1] == expected_label)[0][0]

    ret.append((example_probability, example_rank))
  return ret


def translation_robustness_for_image(model, input_image, expected_label, batch_size=512):
  label_in_model_label_space = expected_label + 1
  with tf.Graph().as_default(), tf.Session() as sess:
    t_positions = tf.placeholder("int32", [None, 2], name='positions')
    t_padded = tf.placeholder("float32", list(input_image.shape), name='padded_image')

    def crop_image(crop_origin):
      t_img = t_padded[crop_origin[0]:(crop_origin[0]+model.image_shape[0]), crop_origin[1]:(crop_origin[1]+model.image_shape[1])]
      return t_img

    cropped_image_t = tf.map_fn(crop_image, t_positions, dtype=t_padded.dtype)

    T = render.import_model(model, cropped_image_t)
    t_logits = T("v0/model/output_logits")
    probabilities_t = tf.nn.softmax(t_logits)[:, label_in_model_label_space]

    y_positions = input_image.shape[0] - model.image_shape[0]
    x_positions = input_image.shape[1] - model.image_shape[1]
    positions = []
    for x_position in range(x_positions):
      for y_position in range(y_positions):
        positions.append((x_position, y_position))
    positions = np.array(positions)
    positions = positions[:1024, ...]

    all_probabilities = []
    for idx in range(0, positions.shape[0], batch_size):
      positions_batch = positions[idx:(idx+batch_size)]
      print(f'running batch of shape: {positions_batch.shape}')

      start_time = time.time()
      batch_probabilities = sess.run(probabilities_t, feed_dict={
        t_positions: positions_batch,
        t_padded: input_image
      })
      all_probabilities.extend(batch_probabilities)
      print(f'operation time: {time.time() - start_time}')

  return all_probabilities


def translation_robustness_for_image_no_map(model, input_image, expected_label, batch_size=512):
  label_in_model_label_space = expected_label + 1
  with tf.Graph().as_default(), tf.Session() as sess:
    t_positions = tf.placeholder("float32", [None, 2], name='positions')
    t_padded = tf.placeholder("float32", list(input_image.shape), name='padded_image')
    t_padded_tiled = tf.tile(t_padded[None], [tf.shape(t_positions)[0], 1, 1, 1])
    t_cropped_images = tf.image.extract_glimpse(t_padded_tiled, model.image_shape[0:2], t_positions, centered=False, normalized=False)

    T = render.import_model(model, t_cropped_images)
    t_logits = T("v0/model/output_logits")
    probabilities_t = tf.nn.softmax(t_logits)[:, label_in_model_label_space]

    y_positions = input_image.shape[0] - model.image_shape[0]
    x_positions = input_image.shape[1] - model.image_shape[1]
    edge_position_y = input_image.shape[0] // 2 - y_positions // 2
    edge_position_x = input_image.shape[1] // 2 - x_positions // 2

    positions = []
    for x_position in range(edge_position_x, edge_position_x + x_positions):
      for y_position in range(edge_position_y, edge_position_y + y_positions):
        positions.append((x_position, y_position))
    positions = np.array(positions)
    positions = positions[:1024, ...]

    all_probabilities = []
    for idx in range(0, positions.shape[0], batch_size):
      positions_batch = positions[idx:(idx+batch_size)]
      print(f'running nomap batch of shape: {positions_batch.shape}')

      start_time = time.time()
      batch_probabilities = sess.run(probabilities_t, feed_dict={
        t_positions: positions_batch,
        t_padded: input_image
      })
      all_probabilities.extend(batch_probabilities)
      print(f'operation time: {time.time() - start_time}')

  return all_probabilities


def translation_robustness_for_image_dense(model, input_image, expected_label, batch_size=512):
  label_in_model_label_space = expected_label + 1
  with tf.Graph().as_default(), tf.Session() as sess:
    # t_positions = tf.placeholder("int32", [None, 2], name='positions')
    t_range = tf.placeholder("int32", [2], name='range')
    t_padded = tf.placeholder("float32", list(input_image.shape), name='padded_image')

    def crop_image(crop_origin):
      t_img = t_padded[crop_origin[0]:(crop_origin[0]+model.image_shape[0]), crop_origin[1]:(crop_origin[1]+model.image_shape[1])]
      # import ipdb; ipdb.set_trace()
      return t_img

    def crop_loop(t_padded, iters):
      def cond(t_padded, i):
        return tf.less(i, iters)

      def body(t_padded, i):
        t_img = t_padded[i:(i+model.image_shape[0]), i:(i+model.image_shape[1])]
        t_img = tf.reshape(t_img, (224, 224, 3))
        return [t_img, tf.add(i, 1)]

      return tf.while_loop(cond, body, [t_padded, iters], shape_invariants=[tf.TensorShape([224, 224, 3]), tf.TensorShape([])])

    # cropped_image_t = tf.map_fn(crop_image, t_positions, dtype=t_padded.dtype)
    cropped_image_t = crop_loop(t_padded, 56)

    T = render.import_model(model, cropped_image_t)
    t_logits = T("v0/model/output_logits")
    probabilities_t = tf.nn.softmax(t_logits)[:, label_in_model_label_space]


    batch_probabilities = sess.run(probabilities_t, feed_dict={
      t_padded: input_image
    })
    import ipdb; ipdb.set_trace()

    all_probabilities = []
    for idx in range(0, positions.shape[0], batch_size):
      positions_batch = positions[idx:(idx+batch_size)]
      print(f'running batch of shape: {positions_batch.shape}')

      batch_probabilities = sess.run(probabilities_t, feed_dict={
        t_positions: positions_batch,
        t_padded: input_image
      })
      all_probabilities.extend(batch_probabilities)

  return all_probabilities


def generate_padded_images(images, padding, verbose=False):
  ret = []
  for i, (sample_image, label, sysnset_id) in enumerate(images):
    input_y = sample_image.shape[0]
    input_x = sample_image.shape[1]
    max_pad_amount = padding
    pad_mode = 'reflect' # 'linear_ramp'  #'reflect'
    padded_image = np.pad(sample_image, [(max_pad_amount, max_pad_amount), (max_pad_amount, max_pad_amount), (0, 0)], pad_mode)

    ret.append((sample_image, label, sysnset_id, padded_image))

  return ret


import lucid.scratch.pretty_graphs.visualizations as vis
from lucid.scratch.pretty_graphs.graph import *

def get_all_layers(model):
  rendered = vis.complete_render_model_graph(model, custom_ops=['DepthwiseConv2dNative'])
  return [n.name for n in rendered["node_boxes"].keys() if n.op not in ["Placeholder", "Softmax", "Matmul"]]

def get_interesting_layers(model):
  layers = get_all_layers(model)
  ret = []
  for layer in layers:
    components = layer.split('/')
    if '/pool_l2' in layer and layer.endswith('/add'):  # capture '*/pool_l2*/add', maybe use regex instead here?
      continue
    if components[-2] == 'fully_connected':
      continue  # skip FC layers for now
    ret.append(layer)

  return ret

def simplified_layer_names(layers):
  last_conv_layer = None
  pool_index = 0
  fc_index = 0

  ret = []
  for layer in layers:
    components = layer.split('/')
    if components[-1].lower() == 'relu':
      if components[-2] == 'Conv' or components[-2].startswith('Conv_'):  # hack to detect strided conv
        clean_name = f'pool_{last_conv_layer}_{pool_index}'
        pool_index += 1
      else:
        clean_name = components[-2]
        last_conv_layer = clean_name
        pool_index = 0
    elif components[-2] == 'fully_connected':
      clean_name = f'fc_{fc_index}'
      fc_index += 1
    elif 'pool' in components[-1].lower():
      clean_name = f'pool_{last_conv_layer}_{pool_index}'
      pool_index += 1
    elif 'blurconv' in components[-1].lower():
      clean_name = f'pool_{last_conv_layer}_{pool_index}'
      pool_index += 1
    elif components[-1].lower() == 'add' or components[-1].lower().startswith('add_'):
      continue
    else:
      raise ValueError(f'unrecognized layer: {layer}')

    ret.append([layer, clean_name])
  return ret



def _inject_shake_layer_into_model(model, t_image, target_layer):
  layers = get_interesting_layers(model)
  clean_layers = simplified_layer_names(layers)
  target_to_shake = [full_name for full_name, simple_name in clean_layers if simple_name == target_layer or full_name == target_layer][0]

  T_base = render.import_model(model, t_image, scope="import_base")
  to_shake_acts = T_base(target_to_shake)
  t_shake_amount = tf.placeholder("float32", [2], name="shake_amount")
  translated_acts = tf.contrib.image.translate(to_shake_acts, t_shake_amount)

  T = render.import_model(model, t_image, input_map={
    target_to_shake: translated_acts
  })

  return T, t_shake_amount


def test_robustness_by_moving_layer(model, input_image, expected_label, target_layer, shake_amount, batch_size=512):
  label_in_model_label_space = expected_label + 1

  with tf.Graph().as_default(), tf.Session() as sess:
    t_positions = tf.placeholder("float32", [None, 2], name='positions')
    t_padded = tf.placeholder("float32", list(input_image.shape), name='padded_image')
    t_padded_tiled = tf.tile(t_padded[None], [tf.shape(t_positions)[0], 1, 1, 1])
    t_cropped_images = tf.image.extract_glimpse(t_padded_tiled, model.image_shape[0:2], t_positions, centered=False, normalized=False)

    T, t_shake_amount = _inject_shake_layer_into_model(model, t_cropped_images, target_layer=target_layer)

    t_logits = T("v0/model/output_logits")
    probabilities_t = tf.nn.softmax(t_logits)[:, label_in_model_label_space]

    y_positions = input_image.shape[0] - model.image_shape[0]
    x_positions = input_image.shape[1] - model.image_shape[1]
    edge_position_y = input_image.shape[0] // 2 - y_positions // 2
    edge_position_x = input_image.shape[1] // 2 - x_positions // 2

    positions = []
    for x_position in range(edge_position_x, edge_position_x + x_positions):
      for y_position in range(edge_position_y, edge_position_y + y_positions):
        positions.append((x_position, y_position))
    positions = np.array(positions)
    positions = positions[:1024, ...]

    all_probabilities = []
    for idx in range(0, positions.shape[0], batch_size):
      positions_batch = positions[idx:(idx+batch_size)]
      print(f'running nomap batch of shape: {positions_batch.shape}')

      start_time = time.time()
      batch_probabilities = sess.run(probabilities_t, feed_dict={
        t_positions: positions_batch,
        t_padded: input_image,
        t_shake_amount: shake_amount
      })
      all_probabilities.extend(batch_probabilities)
      print(f'operation time: {time.time() - start_time}')

  return all_probabilities



def _main():
  cache = shelve.open('translation_robustness.cache')

  if not cache.get('did_nltk_download', False):
    nltk.download('omw')
    cache['did_nltk_download'] = True

  all_models = cache.get('all_models')
  if all_models is None:
    all_models = load_all_models()
    cache['all_models'] = all_models
  print(f'loaded {len(all_models)} models')

  # one time slightly slow (5sec) pull of sample images to use

  model = next(iter(all_models.values()))[0]
  sample_images = cache.get('sample_images')
  if sample_images is None:
    print(f'loading sample images')
    sample_images = imagenet_sample_set(200)

    # find only ones that get classified in top 5 using the 2-2 model
    baseline_classification_results = classification_accuracy(model, [(image_entry[0], image_entry[1]) for image_entry in
                                                                      sample_images])

    sample_images = cache.get('sample_images')
    num_matching_rank = 0

    interesting_sample_images = []
    for i, (sample_image, label, sysnset_id) in enumerate(sample_images):
      if baseline_classification_results[i][1] < 5:
        num_matching_rank += 1
        interesting_sample_images.append(sample_images[i])

    print(f'images with correct top-5 classification: {num_matching_rank} / {len(sample_images)} (using rank)')
    sample_images = interesting_sample_images[:100]
    cache['sample_images'] = sample_images

  padded_images = generate_padded_images(sample_images, 28)

  # robustness = translation_robustness_for_image_no_map(model, padded_images[0][-1], padded_images[0][1], batch_size=128)
  # robustness_map = translation_robustness_for_image(model, padded_images[0][-1], padded_images[0][1], batch_size=128)
  # assert robustness == robustness_map

  baseline_probs = test_robustness_by_moving_layer(model, padded_images[0][-1], padded_images[0][1], target_layer='pool_4e_0', shake_amount=[0, 0], batch_size=128)
  shifted_probs = test_robustness_by_moving_layer(model, padded_images[0][-1], padded_images[0][1], target_layer='pool_4e_0', shake_amount=[1, 1], batch_size=128)

  import ipdb; ipdb.set_trace()


if __name__ == '__main__':
  _main()
