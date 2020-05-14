from os import listdir
import numpy as np
from pickle import dump
import keras
import string
from PIL import Image

START_TOKEN = "START"
STOP_TOKEN = "STOP"
UNK = "*UNK*"

'''
	PREPROCESSING.

	1) Use VGG to get features of all the images.
		a) Import a keras VGG16, and exclude the final image classification layer.
		b) Create a new Keras model with the inputs being those of VGG, and outputs being the 
			new final layer of the vgg: the image feature matrix. 
		c) Create a dictionary where id: photo_id --> value: feature matrix
		d) Get features for each image in folder.
	2) Convert the textual caption information into:
		a) vocabulary
		b) a dictionary of cleaned (remove punct, lowercase, etc.) captions where:
			id: image_id --> value: list: [captions]
    3) Save everything to a files!! 
'''

# IMAGE PREPROCESSING
model = keras.applications.vgg16.VGG16()
model = keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
features = {}

directory = 'flickr30k_images'
total = len(listdir(directory))

print("Progress:")
for i, name in enumerate(listdir(directory)):
	img_path = directory + '/' + name
	image_id = name.split('.')[0]

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	x = keras.preprocessing.image.img_to_array(img)

	# The keras preprocess image function needs the input to have batch_size as first
	# argument, so we just add b_sz = 1
	x = np.expand_dims(x, axis=0)

	# Since we're using VGG weights, need to run this to transform the current img
	# into a format that is compatible with the VGG model.
	x = keras.applications.vgg16.preprocess_input(x)
	
	image_feature = model.predict(x, verbose=0) # feature will have shape (1, 4096)

	features[image_id] = image_feature

	# This can take a long time, so this will let you know how much longer
	# you have to go.
	
	if i%100 == 0:
		print(str(i) + " / " + str(total))

print('Number of Extracted Features: %d' % len(features))

dump(features, open('features.pkl', 'wb')) # save the features dict to a file so you never have to do this again

# TEXT PREPROCESSING 

with open('Flickr8k.token.txt','r') as f:
    text = f.read()

# remove punctuation with s.translate(str.maketrans('', '', string.punctuation)) thx stackoverflow

captions = {}
vocabulary = set()

lines = text.split('\n')
for line in lines:
	# print(line)
	if len(line) < 2: continue
	words = line.split()
	img_id = words[0].split('.')[0]
	# print(img_id)
	caption = words[1:]
	
	'''
		clean the caption:
			1) lowercase
			2) remove punct
			3) remove digits
			4) remove single letter "words" (ex. I, a)

	'''
	caption = [x.lower() for x in caption]
	caption = [x.translate(str.maketrans('', '', string.punctuation)) for x in caption]
	caption = [x for x in caption if x.isalpha()]
	caption = [x for x in caption if len(x) > 1]

	# add the words to the vocab if they're not in there already
	[vocabulary.add(x) for x in caption]
	
	caption = ' '.join(caption) # this should be a string for training
	caption = START_TOKEN + ' ' +  caption + ' ' + STOP_TOKEN
	if img_id not in captions:
		captions[img_id] = [caption]
	else: captions[img_id].append(caption)
	
print('Vocabulary Size: %d' % len(vocabulary))

# save the cleaned captions to an output file
lines = []
for key, value in captions.items():
	for description in value:
		lines.append(key + ' ' + description)
data = '\n'.join(lines)

# with open('captions.txt','w') as f:
# 	f.write(data)

dump(captions, open('captions.pkl', 'wb'))
dump(vocabulary, open('vocab.pkl', 'wb')) #why not


