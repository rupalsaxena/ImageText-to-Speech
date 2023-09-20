import pandas as pd
import numpy as np

from glob import glob
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from PIL import Image

plt.style.use('ggplot')


img_filenames = glob('../input/textocr-text-extraction-from-images-dataset/train_val_images/train_images/*')
annot = pd.read_parquet('../input/textocr-text-extraction-from-images-dataset/annot.parquet')
import pdb; pdb.set_trace()