import config 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam


from sklearn.preprocessing import LabelBinarizer
import numpy as np 



cfg = config.Config()



test_data = []
test_imagePaths = []
test_labels = []

# you can change the annotation file here
test_annotations = open('dataset/test.txt', 'r')



model = load_model(cfg.MODEL_PATH)


for f in test_annotations : 
    annotation = f.split()
    basename = annotation[0]
    startX = float(annotation[1])
    startY = float(annotation[2])
    endX = float(annotation[3])
    endY = float(annotation[4])
    label = int(annotation[5])
    image_path = cfg.TEST_PATH + '/' + basename 
    # load the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # update our list of data, class labels, and
	# image paths
    test_data.append(image)
    test_labels.append(label)
    test_imagePaths.append(image_path)
test_annotations.close()


test_data = np.array(test_data, dtype="float32") / 255.0
test_lb = LabelBinarizer()
test_labels = test_lb.fit_transform(test_labels)


# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head 
losses = {
	"class_label": "categorical_crossentropy",
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


# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
	"class_label": test_labels,
}

print('\n\n')
print("Evaluate on test data")
print('\n')

# The results of the test
results = model.evaluate(test_data, testTargets, batch_size=20)
print("test loss, test acc:", results)