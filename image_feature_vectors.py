import tensorflow as tf
import tensorflow_hub as hub

# For saving 'feature vectors' into a txt file
import numpy as np

# Time for measuring the process time
import time

# Glob for reading file names in a folder
import glob
import os.path

def load_img(path,model):

  # Reads the image file and returns data type of string
  img = tf.io.read_file(path)

  # Decodes the image to W x H x 3 shape tensor with type of uint8
  img = tf.io.decode_jpeg(img, channels=3)
  if model==1:
  # Resize the image to 224 x 244 x 3 shape tensor
    img = tf.image.resize_with_pad(img, 224, 224)
  else:
    img = tf.image.resize_with_pad(img, 380, 380)
  # Converts the data type of uint8 to float32 by adding a new axis
  # This makes the img 1 x 224 x 224 x 3 tensor with the data type of float32
  # This is required for the mobilenet model we are using
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

  return img


def get_image_feature_vectors():

  i = 0

  start_time = time.time()

  print("---------------------------------")
  print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Started at %s" %time.ctime())
  print("---------------------------------")

  # Definition of module with using tfhub.dev handle
  module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4" 
  module_handle2='https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1'
  # Load the module
  module = hub.load(module_handle)
  module2 =hub.load(module_handle2)
  print("---------------------------------")
  print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Completed at %s" %time.ctime())
  print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))

  print("---------------------------------")
  print ("Step.2 of 2 - Generating Feature Vectors -  Started at %s" %time.ctime())
 

  # Loops through all images in a local folder
  for filename in glob.glob('/home/ImageSimilarity/test1/*.jpg'): #assuming gif
    i = i + 1

    print("-----------------------------------------------------------------------------------------")
    print("Image count                     :%s" %i)
    print("Image in process is             :%s" %filename)

    # Loads and pre-process the image
    img = load_img(filename,1)
    img2=load_img(filename,2)

    # Calculate the image feature vector of the img
    features = module(img)   
    features2=module2(img2)
  
    # Remove single-dimensional entries from the 'features' array
    feature_set = np.squeeze(features)  
    feature_set2= np.squeeze(features2)
    # Saves the image feature vectors into a file for later use

    outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
    out_path = os.path.join('/home/ImageSimilarity/test1/', outfile_name)
    out_path2=os.path.join('/home/ImageSimilarity/test2/', outfile_name)
    # Saves the 'feature_set' to a text file
    np.savetxt(out_path, feature_set, delimiter=',')
    np.savetxt(out_path2, feature_set2, delimiter=',')
    print("Image feature vector saved to   :%s" %out_path)
  
  print("---------------------------------")
  print ("Step.2 of 2 - Generating Feature Vectors - Completed at %s" %time.ctime())
  print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
  print("--- %s images processed ---------" %i)
    
get_image_feature_vectors()
