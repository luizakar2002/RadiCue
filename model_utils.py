import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import skimage, skimage.io

import requests
from PIL import Image
import cv2

from grad import grad_cam

labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

sns.reset_defaults()

def get_mean_std_per_batch(df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        path = IMAGE_DIR + img
        sample_data.append(np.array(image.load_img(path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std  
  
def load_image(path, preprocess=True, H = 320, W = 320):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        mean, std = get_mean_std_per_batch(df, H=H, W=W)
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

def load_image_normalize(url, mean=124.463743359375, std=63.81871748655583, H=320, W=320):
    response = requests.get(url)
    if response.status_code == 200:
        file_name = 'saved_locally.png'
        with open(file_name, 'wb') as file:
            file.write(response.content)
            print(f"Image saved as {file_name}")
    else:
        print("Failed to download the image.")
    path = 'saved_locally.png'
    x = image.load_img(path, target_size=(H, W))
    x -= np.array([mean])
    x /= std
    x = np.expand_dims(x, axis=0)
    return x

def predictFromModel(url):
    # im = skimage.io.imread(url)
    # im_path = '00005410_000.png' # mass
    print("URL")
    print(url)
    im = load_image_normalize(url)


    # loaded_model = tf.keras.models.load_model('model.h5', compile=False)

    # predictions = loaded_model.predict(im)
    # if np.max(predictions[0]) > 0.5:
    #     index_of_max = np.argmax(predictions[0])
    #     predicted_label = labels[np.argmax(predictions[0])]
    #     cam = grad_cam(loaded_model, im, index_of_max, 'conv5_block16_concat')
    #     plt.imshow(load_image(im_path, preprocess=False), cmap='gray')
    #     plt.imshow(cam, cmap='magma', alpha=0.5)
    #     plt.axis('off')
    #     buffer = io.BytesIO()
    #     plt.savefig(buffer, format='png')
    #     buffer.seek(0)
    #     vis_img = buffer.getvalue()
    #     # return vis_img, predicted_label
    #     return predicted_label
    # # return im, 'Normal'
    return 'Normal'







# im_path = '00008270_015.png' 
# im_path = '00005410_000.png' # mass
# im = load_image_normalize(im_path)


# loaded_model = tf.keras.models.load_model('model.h5', compile=False)

# predictions = loaded_model.predict(im)
# print(predictions)
# if np.max(predictions[0] > 0.5):
#     index_of_max = np.argmax(predictions[0])
#     predicted_label = labels[np.argmax(predictions[0])]
#     cam = grad_cam(loaded_model, im, 5, 'conv5_block16_concat')
#     plt.imshow(load_image(im_path, preprocess=False), cmap='gray')
#     plt.imshow(cam, cmap='magma', alpha=0.5)
#     plt.axis('off')
#     plt.show()