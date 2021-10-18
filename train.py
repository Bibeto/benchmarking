import config
import argparse

from tensorflow.keras.layers import Dense, Flatten
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
import numpy as np
import os
import seed


# Command line interface 
# to make training easier through command line only
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--learning-rate", required=False, type=float, default=1e-5, 
	help="input the learning rate for the training")
parser.add_argument("-e", "--epochs", required=False, type=int, default=50, 
	help="input the number of epochs for the training")
parser.add_argument("-b", "--batch-size", required=False, type=int, default=32,  
	help="input the batch size for the training")
parser.add_argument("-s", "--seed", required=False, type=int, default=50, 
	help="change the initial parameters for the training through fixing a new seed")
parser.add_argument("-f", "--fix-seed", required=False, type=bool, default=True, 
	help="fix the seed or not True for yes / False for no")
args = vars(parser.parse_args())

# initialize a configuration file 
cfg = config.Config(lr=args["learning_rate"], epochs=args["epochs"], 
    batch_size=args["batch_size"], seed=args["seed"], image_size=224, number_classes=50)

# Setting a seed to make reproducible results 
seed.fix_seed(cfg.SEED, fix_it=args["fix_seed"])



## initialize the list of data (images), class labels, target bounding
## box coordinates, and image paths
print("[INFO] loading dataset...")

train_data = []
validation_data = []

train_imagePaths = []
validation_imagePaths = []

train_labels = []
validation_labels = []

# loading the annotation files
train_annotations = open(cfg.TRAIN_ANNOTATIONS_PATH, 'r')
validation_annotations = open(cfg.VALIDATION_ANNOTATIONS_PATH, 'r')



for f in train_annotations : 
    annotation = f.split()
    basename = annotation[0]
    label = int(annotation[5])
    image_path = cfg.TRAIN_PATH + '/' + basename 
    # load the image 
    image = load_img(image_path)
    image = img_to_array(image)

    # update our list of data, class labels, and
	# image paths
    train_data.append(image)
    train_labels.append(label)
    train_imagePaths.append(image_path)
train_annotations.close()


for f in validation_annotations : 
    annotation = f.split()
    basename = annotation[0]
    label = int(annotation[5])
    image_path = cfg.VALIDATION_PATH + '/' + basename 

    image = load_img(image_path)
    image = img_to_array(image)

    # update our list of data, class labels, and
	# image paths
    validation_data.append(image)
    validation_labels.append(label)
    validation_imagePaths.append(image_path)
validation_annotations.close()

# convert the data, class labels, and image paths to
# NumPy arrays, scaling the input pixel intensities from the range
# [0, 255] to [0, 1]
train_data = np.array(train_data, dtype="float32") / 255.0
train_labels = np.array(train_labels, dtype='int8')
train_imagePaths = np.array(train_imagePaths)

validation_data = np.array(validation_data, dtype="float32") / 255.0
validation_labels = np.array(validation_labels, dtype='int8')
validation_imagePaths = np.array(validation_imagePaths)


# perform one-hot encoding on the labels
train_lb = LabelBinarizer()
train_labels = train_lb.fit_transform(train_labels)

validation_lb = LabelBinarizer()
validation_labels = validation_lb.fit_transform(validation_labels)

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGGFace(include_top=False, pooling='avg')
# freeze all VGG layers so they will *not* be updated during the
# training process

vgg.trainable = False
model_ouput = vgg.output

model_ouput = Flatten(name='flatten')(model_ouput)
model_ouput = Dense(4096, name='fc6' ,activation='relu')(model_ouput)
model_ouput = Dense(4096, name='fc7', activation='relu')(model_ouput)
model_ouput = Dense(len(train_lb.classes_), activation="softmax",name="class_label")(model_ouput)

# put together our model which accept an input image and then output
model = Model(
	inputs=vgg.input,
	outputs=model_ouput)



# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head 
losses = {
	"class_label": "categorical_crossentropy"
}
# define a dictionary that specifies the weights per loss 
lossWeights = {
	"class_label": 1.0,
}
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=cfg.INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())


# construct a dictionary for our target training outputs
trainTargets = {
	"class_label": train_labels
}
# construct a second dictionary, this one for our target testing
# outputs
validationTargets = {
	"class_label": validation_labels
}


# train the network for class label
# prediction
print('\n\n')
print("[INFO] training model...")

H = model.fit(
	train_data, trainTargets,
	validation_data=(validation_data, validationTargets),
	batch_size=cfg.BATCH_SIZE,
	epochs=cfg.NUM_EPOCHS,
	verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(cfg.MODEL_PATH, save_format="h5")



print("[INFO] saving accuracy, loss curves...")
N = np.arange(0, cfg.NUM_EPOCHS)

## Plotting the accuracy/loss of both the training 
## dataset and validation dataset 

# plotting the accuracy
plt.style.use("ggplot")
plt.plot(N, H.history["accuracy"], label="accuracy")
plt.plot(N, H.history['val_accuracy'], label="val_accuracy")
plt.title("Model Accuracy Curve")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy"]) 
plotPath = os.path.sep.join([cfg.PLOTS_PATH, "accs.png"])
plt.savefig(plotPath)
plt.close()

# Plotting the loss
plt.plot(N, H.history['loss'], label="Loss")
plt.plot(N, H.history['val_loss'], label="val_loss")
plt.title("Model Loss Curve")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"]) 
plotPath = os.path.sep.join([cfg.PLOTS_PATH, "losses.png"])
plt.savefig(plotPath)