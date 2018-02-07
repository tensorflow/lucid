"""Neural Image Style Transfer utils."""

import tensorflow as tf


def gram_style(a):
  with tf.name_scope('gram'):
    chn = int(a.shape[-1])
    a = tf.reshape(a, [-1, chn])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def mean_l1_loss(g1, g2):
  with tf.name_scope('mean_l1_loss'):
    return tf.reduce_mean(tf.abs(g1-g2))


class StyleLoss(object):
  """Image Style Loss.
  
  A variant of style component of Artistic Image Style Transfer loss,
  mainly inspired by [1].

  [1] Leon A. Gatys et al. "A Neural Algorithm of Artistic Style"
      https://arxiv.org/abs/1508.06576
  """

  def __init__(self, style_layers, ema_decay=0.0,
               style_func=gram_style,
               loss_func=mean_l1_loss):
    """Initilize style loss.
    
    Args:
      style_layers: List of tensors that are used to compute statistics that
        define a style.
      ema_decay: Number in range [0.0 .. 1.0]. Loss function is computed against
        moving averaged versions of style statistics if ema_decay > 0.0.
        This is useful when each optimisation step only covers some part of
        the full output image.
      style_func: Function that is used to compute layer statistics.
      loss_func: Function that is used to compute difference between two
        outputs of 'style_func'.
    """
    self.input_grams = [style_func(s) for s in style_layers]
    if ema_decay:
      ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
      update_ema_op = ema.apply(self.input_grams)
      with tf.control_dependencies([update_ema_op]):
        self.effective_grams = [g + tf.stop_gradient(ema.average(g)-g)
                           for g in self.input_grams]
    else:
      self.effective_grams = self.input_grams
    
    self.target_vars = [tf.Variable(tf.zeros_like(g), trainable=False)
                        for g in self.input_grams]
    self.style_losses = [loss_func(g, gv)
                         for g, gv in zip(self.effective_grams, self.target_vars)]
    self.style_loss = tf.add_n(self.style_losses)

  def set_style(self, input_feeds):
    """Set target style variables.
    
    Expected usage: 
      style_loss = StyleLoss(style_layers)
      ...
      init_op = tf.global_variables_initializer()
      init_op.run()
      
      feeds = {... session.run() 'feeds' argument that will make 'style_layers'
               tensors evaluate to activation values of style image...}
      style_loss.set_style(feeds)  # this must be called after 'init_op.run()'
    """
    sess = tf.get_default_session()
    computed = sess.run(self.input_grams, input_feeds)
    for v, g in zip(self.target_vars, computed):
      v.load(g)
