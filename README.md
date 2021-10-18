# Copying Code
Clone the repository in your machine : 

git clone https://github.com/Bibeto/benchmarking


# Dependencies 
To install the dependencies using pip : 
pip install -r requirements.txt

To add your custom dataset for training : 

cp <your_training_dataset> dataset/train

cp <your_testing_dataset> dataset/test

<your_validation_dataset> dataset/validation



add your annotation files train.txt, test.txt and validation.txt 
where the annotation file is with the form : 
image_name startX startY endX endY class



# Training
To train the model type in your terminal : 
python train.py 
you can type : python train.py -h 
to see the different parameters you can change in the training 



To give credit to its proper owners, the models used are from: 

the VGG16 : https://github.com/rcmalli/keras-vggface

Squeezenet : https://github.com/DT42/squeezenet_demo
