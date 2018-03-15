import lucid.optvis.param.spatial as spatial
import lucid.optvis.param.color as color

def image_sample(shape, decorrelate=True, sd=None, decay_power=1):
  raw_spatial = rand_fft_image(shape, sd=sd, decay_power=decay_power)
  return color.to_valid_rgb(raw_spatial, decorrelate=decorrelate)

def rand_fft_image(shape, sd=None, decay_power=1):
  b, h, w, ch = shape
  sd = 0.01 if sd is None else sd
  
  imgs = []
  for _ in range(b):
    freqs = spatial._rfft2d_freqs(h, w)
    fh, fw = freqs.shape
    spectrum_var = sd*tf.random_normal([2, ch, fh, fw], dtype="float32")
    spectrum = tf.complex(spectrum_var[0], spectrum_var[1])
    spertum_scale = 1.0 / np.maximum(freqs, 1.0/max(h, w))**decay_power
    # Scale the spectrum by the square-root of the number of pixels
    # to get a unitary transformation. This allows to use similar
    # leanring rates to pixel-wise optimisation.
    spertum_scale *= np.sqrt(w*h)
    scaled_spectrum = spectrum * spertum_scale
    img = tf.spectral.irfft2d(scaled_spectrum)
    # in case of odd input dimension we cut off the additional pixel
    # we get from irfft2d length computation
    img = img[:ch, :h, :w]
    img = tf.transpose(img, [1, 2, 0])
    imgs.append(img)
  return tf.stack(imgs)/4.
