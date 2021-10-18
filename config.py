import os

from tensorflow.keras.preprocessing import image

class Config: 
    def __init__(self, lr=1e-5, epochs=50, batch_size=32, seed=50, image_size=224, number_classes=50): 
        # define the base path to the input dataset and then use it to derive
        # the path to the input images and annotation CSV files
        self.BASE_PATH = "dataset"
        self.TRAIN_PATH = os.path.sep.join([self.BASE_PATH, "train"])
        self.TEST_PATH =  os.path.sep.join([self.BASE_PATH, "test"])
        self.VALIDATION_PATH =  os.path.sep.join([self.BASE_PATH, "validation"])

        self.TRAIN_ANNOTATIONS_PATH = os.path.sep.join([self.BASE_PATH, "train.txt"])
        self.VALIDATION_ANNOTATIONS_PATH = os.path.sep.join([self.BASE_PATH, "validation.txt"])
        self.TEST_ANNOTATIONS_PATH = os.path.sep.join([self.BASE_PATH, "test.txt"])

        # define the path to the base output directory
        BASE_OUTPUT = "output"
        # define the path to the output model, label binarizer, plots output
        # directory, and testing image paths
        self.MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
        self.LB_PATH_TRAIN = os.path.sep.join([BASE_OUTPUT, "lb_train.pickle"])
        self.LB_PATH_VALIDATION = os.path.sep.join([BASE_OUTPUT, "lb_validation.pickle"])
        self.LB_PATH_TEST = os.path.sep.join([BASE_OUTPUT, "lb_test.pickle"])
        self.PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])


        # initialize our initial learning rate, number of epochs to train
        # for, and the batch size
        self.INIT_LR = lr
        self.NUM_EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.SEED = seed
        self.IMAGE_SIZE = image_size
        self.NUMBER_CLASSES = number_classes