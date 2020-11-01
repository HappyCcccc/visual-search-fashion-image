# Numpy for loading image feature vectors from file
import numpy as np

# Time for measuring the process time
import time

# Glob for reading file names in a folder
import glob
import os.path

# json for storing data in json file
import json

# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial


def get_target(dir):
    print('in target')
    f=open(dir, "r")
    a=f.read()
    return a.split(':')[1].split('.')[0]


res1,res2=[],[]


def cluster(dir,res):

  start_time = time.time()
  
  print("---------------------------------")
  print ("Step.1 - ANNOY index generation - Started at %s" %time.ctime())
  print("---------------------------------")

  # Defining data structures as empty dict
  file_index_to_file_name = {}
  file_index_to_file_vector = {}
  #file_index_to_product_id = {}
  file_to_file_index={}
  # Configuring annoy parameters
  dims = 1792
  n_nearest_neighbors = 20
  trees = 10000

  # Reads all file names which stores feature vectors 
  allfiles = glob.glob(dir)
  #allfiles = glob.glob('/home/ImageSimilarity/test1/*.npz')
  #allfiles2=glob.glob('/home/ImageSimilarity/test2/*.npz')
  t = AnnoyIndex(dims, metric='angular')

  for file_index, i in enumerate(allfiles):
    
    # Reads feature vectors and assigns them into the file_vector 
    file_vector = np.loadtxt(i)

    # Assigns file_name, feature_vectors and corresponding product_id
    file_name = os.path.basename(i).split('.')[0]
    file_index_to_file_name[file_index] = file_name
    file_index_to_file_vector[file_index] = file_vector

    # Adds image feature vectors into annoy index   
    t.add_item(file_index, file_vector)

    print("---------------------------------")
    print("Annoy index     : %s" %file_index)
    print("Image file name : %s" %file_name)
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
  
  for key in file_index_to_file_name:
    file_to_file_index[file_index_to_file_name[key]]=key
  # Builds annoy index
  t.build(trees)

  print ("Step.1 - ANNOY index generation - Finished")
  print ("Step.2 - Similarity score calculation - Started ") 
  
  named_nearest_neighbors = []
  print('test')
  i=file_to_file_index[get_target('/home/ImageSimilarity/data.cfg')]

  master_file_name = file_index_to_file_name[i]
  master_vector = file_index_to_file_vector[i]

    # Calculates the nearest neighbors of the master item
  nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

    # Loops through the nearest neighbors of the master item
  for j in nearest_neighbors:

    print(j)

      # Assigns file_name, image feature vectors and product id values of the similar item
    neighbor_file_name = file_index_to_file_name[j]
    neighbor_file_vector = file_index_to_file_vector[j]
    #neighbor_product_id = file_index_to_product_id[j]

      # Calculates the similarity score of the similar item
    similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
    rounded_similarity = int((similarity * 10000)) / 10000.0

      # Appends master product id with the similarity score 
      # and the product id of the similar items
    if rounded_similarity > 0.1:
        named_nearest_neighbors.append({
          'similarity': rounded_similarity,
          'master_pi': master_file_name,
          'similar_pi': neighbor_file_name})
        
        res.append({
          'similarity': rounded_similarity,
          'master_pi': master_file_name,
          'similar_pi': neighbor_file_name})


  print ("Step.3 - Data stored in 'nearest_neighbors.json' file ") 
  print("--- Prosess completed in %.2f minutes ---------" % ((time.time() - start_time)/60))

cluster('/home/ImageSimilarity/test1/*.npz',res1)
cluster('/home/ImageSimilarity/test2/*.npz',res2)

print(res1)
print(res2)
res_map={}
for r in res1:
  res_map[r['similar_pi']]=[r['similarity']]
for r in res2:
  if r['similar_pi'] in res_map:
    res_map[r['similar_pi']].append(r['similarity'])
for key in res_map:
  avg=(res_map[key][0]+res_map[key][1])/2
  res_map[key].append(avg)
print(res_map)
with open('nearest_neighbors.json', 'w') as out:
    json.dump(res_map, out)
