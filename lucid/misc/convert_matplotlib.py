import io
import numpy as np
from PIL import Image

def matplotlib_to_numpy(plt):
  """Convert a matplotlib plot to a numpy array represent it as an image.

  Inputs:
    plot - matplotlib plot

  Returns:
    A numpy array with shape [W, H, 3], representing RGB values between 0 and 1.
  """
  
  f = io.BytesIO()
  plt.savefig(f, format="png")
  f.seek(0)
  arr = np.array(Image.open(f)).copy()
  f.close()
  return arr/255.
