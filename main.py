import sys
import os
import numpy as np
from pickle import load, dump
import keras
import string
from nltk.translate.bleu_score import corpus_bleu

# I also saved the vocab lollll 
vocabulary = load(open("vocab.pkl", 'rb'))
# print(list(vocabulary)[:5])

START_TOKEN = "START"
STOP_TOKEN = "stop"
UNK = "*UNK*"
BATCH_SIZE = 20
VOCAB_SIZE = len(vocabulary)
max_caption_length = 0
WINDOW_SIZE = max_caption_length
num_epochs = 1
mode = sys.argv[1]

'''
	TRAINING:
	
	1) Choose the images that you will train on.
	2) Get the training images' captions and features.
		a) Add start and stop tokens to each description
		b) Convert the captions to word2id vectors
	3) 


'''
def model_definition(vocab_size, max_length):
	'''
		First part is the mini model that uses the image features.
	'''
	image_inputs = keras.layers.Input(shape=(4096,))
	dropout = keras.layers.Dropout(0.5)(image_inputs)
	layer1 = keras.layers.Dense(256, activation='relu')(dropout)
	'''
		Second part is the mini model that uses the text/caption features.
	'''
	text_inputs = keras.layers.Input(shape=(max_length,))
	embs = keras.layers.Embedding(vocab_size, 256, mask_zero=True)(text_inputs)
	dropout_text = keras.layers.Dropout(0.5)(embs)
	layer2 = keras.layers.LSTM(256)(dropout_text)
	'''
		Add the outputs of the image layer and the caption layer into
		the decoder, then pass through dense layer with ReLu activation, 
		then pass through another dense layer for a softmax prediction over 
		all the vocab for the next word in the outputted sequence caption.
	'''
	decoder1 = keras.layers.merge.add([layer1, layer2])
	decoder2 =keras.layers. Dense(256, activation='relu')(decoder1)
	output = keras.layers.Dense(vocab_size, activation='softmax')(decoder2)
	'''
		Build the model with the above architecture. For conceptuals:
		inputs = [image_features, caption]
		output = word
	'''
	model = keras.models.Model(inputs=[image_inputs, text_inputs], outputs=output)
	'''
		Minimize the model's cross-entropy loss with the Adam Optimizer, and 
		the default learning rate.
	'''
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model

'''
	Each batch:
		input1 is the image features
		input2 is the textual caption for the image
			ex. 'START Dog runs in the water'
		output is the textual caption for the image but for decoder
			ex. 'Dog runs in the water STOP'

'''
def make_batch(tokenizer, max_caption_length, captions_list, image):
	input1 = []
	input2 = []
	output = []
	for caption in captions_list:
		sequence = tokenizer.texts_to_sequences([caption])[0]
		for i in range(1, len(sequence)):
			in_, out_ = sequence[:i], sequence[i]
			in_ = keras.preprocessing.sequence.pad_sequences([in_], maxlen=max_caption_length)[0]
			out_ = keras.utils.to_categorical([out_], num_classes=VOCAB_SIZE)[0]
			input1.append(image)
			input2.append(in_)
			output.append(out_)
	return np.array(input1), np.array(input2), np.array(output)

def data_generator(captions, images, tokenizer, max_caption_length):
	while True:
		for image_id, captions_list in captions.items():
			image = images[image_id][0]
			encoder_input, decoder_input, decoder_output = make_batch(tokenizer, max_caption_length, captions_list, image)
			yield [[encoder_input, decoder_input], decoder_output]

# def make_batch(data, batch_size=BATCH_SIZE):
# 	index = 0
# 	try:
# 		while True:
# 			batch = []
# 			label_batch = []
# 			lengths = []
			
# 			for i in range(batch_size):
# 				current = []
# 				current_label = []

# 				for j in range(WINDOW_SIZE):
# 					current.append(vocabulary[data[index]])
# 					current_label.append(vocabulary[data[index + 1]])
# 					index += 1

# 				batch.append(current)
# 				label_batch.append(current_label)
# 				lengths.append(WINDOW_SIZE)

# 			yield batch, label_batch, lengths
# 	except IndexError:
# 		return


# with open('Flickr_8k.trainImages.txt', 'r') as f:
# 	trainImages = f.read()

# with open('Flickr_8k.testImages.txt', 'r') as f:
# 	testImages = f.read()

with open('Flickr8k_text/Flickr_8k.trainImages.txt', 'r') as f:
	trainImages = f.read()

with open('Flickr8k_text/Flickr_8k.testImages.txt', 'r') as f:
	testImages = f.read()

# we need to get the image id's from this:
train_img_names = []
for line in trainImages.split('\n'):
	train_img_names.append(line.split('.')[0])
train_img_names = set(train_img_names)

test_img_names = []
for line in testImages.split('\n'):
	test_img_names.append(line.split('.')[0])
test_img_names = set(test_img_names)


# get features of these training images
features = load(open("features.pkl", 'rb'))
train_img_features = {k: features[k] for k in train_img_names if not(k == '')}
test_img_features = {k: features[k] for k in test_img_names if not(k == '')}

# get captions for each of these training images
captions = load(open("captions.pkl", 'rb'))
# for key in list(captions)[:5]: print(key, captions[key])
train_captions = {k: captions[k] for k in train_img_names if not(k == '')}
test_captions = {k: captions[k] for k in test_img_names if not(k == '')}

# convert to word id's (not used anymore)
# i = 2
# word2id = {"START": 0, "STOP": 1}
# for word in vocabulary:
# 	if word not in word2id:
# 		word2id[word] = i
# 		i += 1

# make a list of captions from the dict for input into tokenizer
# flatten the list
captions_list = [caption for captions in train_captions.values() for caption in captions]
print(captions_list[:50])
max_caption_length = max(len(s.split()) for s in captions_list)


tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(captions_list)
# save the tokenizer for easy image generating
dump(tokenizer, open('tokenizer.pkl', 'wb'))

VOCAB_SIZE = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % VOCAB_SIZE)

'''
	Create the model for training,
	Generate the batched data using the tokenizer
	Train the model on the batches.
	Save the model ! --> We will use it to generate captions.
'''
if mode == 'train':
	model = model_definition(VOCAB_SIZE, max_caption_length)
	steps = len(train_img_features)

	generator = data_generator(train_captions, train_img_features, tokenizer, max_caption_length)
	model.fit_generator(generator, epochs=num_epochs, steps_per_epoch=len(train_img_features), verbose=1)
	model.save('my_model.h5')
	print("Done training.")

'''
	TESTING:
	Evaluate the model on the testing data
	Starting with START token, predict each consecute
	word in the caption sequence one by one by choosing the 
	one with the highest probability. End predicition once you
	have predicted the STOP token.
'''
if mode == 'test':
	print("Loading model...")
	model = keras.models.load_model('my_model.h5')
	test_captions_list = [caption for captions in test_captions.values() for caption in captions]

	def predict_caption(model, tokenizer, image, max_caption_length):
		def word2id(num, tokenizer):
			for word, i in tokenizer.word_index.items():
				if i == num:
					return word
			return None
		prediction = START_TOKEN
		for _ in range(max_caption_length):
			sequence = tokenizer.texts_to_sequences([prediction])[0]
			sequence = keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_caption_length)
			yhat = model.predict([image,sequence], verbose=0)
			yhat = np.argmax(yhat)
			word = word2id(yhat, tokenizer)
			if word is None:
				break
			prediction += ' ' + word
			if word == STOP_TOKEN:
				break
		return prediction

	actual, predicted = [], []

	print("Making caption predictions...")
	tot = len(test_captions.keys())
	i = 0
	for image_id, captions_list in test_captions.items():
		yhat = predict_caption(model, tokenizer, test_img_features[image_id], max_caption_length)
		true_labels = [x.split() for x in captions_list]
		actual.append(true_labels)
		predicted.append(yhat.split())
		
		if i%100 == 0:
			print(str(i) + " / " + str(tot))
		i += 1

	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
