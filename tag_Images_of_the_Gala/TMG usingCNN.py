from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation,Dropout,BatchNormalization
from keras import regularizers,optimizers
from keras import backend
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy
import pandas

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''

    # def on_epoch_end(self, epoch, logs={}):
    #     self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
    #     '{:.0f}'.format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
    #     self.epoch_id += 1

    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'


script_dir = os.path.dirname('__file__')
training_set_path = os.path.join(script_dir, r'E:\course\hellow ml !\data set\9d34462453e311ea\dataset\Train_images')
test_set_path = os.path.join(script_dir, r'E:\course\hellow ml !\data set\9d34462453e311ea\dataset\Test_images')
'''
#Initilizing the CNN
classifier=Sequential()
 #step 1. Convolution
classifier.add(Convolution2D(32,(3,3), input_shape=(32,32,3),activation='relu'))
#32 replaced by 64 or128 or256 and 64 is replaced by 256 or 512 or more...

# step2. pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a third convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#step3. Flatten
classifier.add(Flatten())

#step4. Full Connection
classifier.add(Dense(output_dim=512,activation='relu'))
classifier.add(Dense(output_dim=0.5,activation='softmax'))
# Compiling the CNN
classifier.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
'''

classifier = Sequential()
classifier.add(Convolution2D(64, (3, 3), padding='same',input_shape=(64,64,3)))
classifier.add(Activation('relu'))
classifier.add(Convolution2D(64, (3, 3)))
classifier.add(Activation('relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Convolution2D(128, (3, 3), padding='same'))
classifier.add(Activation('relu'))
classifier.add(Convolution2D(128, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
classifier.add(Flatten())
classifier.add(Dense(512))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(4, activation='softmax'))
classifier.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

#part2. fitting the CNN to the image
# import pandas
# def append_ext(fn):
#     return fn+".png"

train_df = pandas.read_csv(r"E:\course\hellow ml !\data set\9d34462453e311ea\dataset\train.csv",dtype=str)
test_df = pandas.read_csv(r"E:\course\hellow ml !\data set\9d34462453e311ea\dataset\test.csv",dtype=str)

# train_df["Image"]=train_df["Image"].apply(append_ext)
# test_df["Image"]=test_df["Image"].apply(append_ext)

train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.25)

test_datagen = ImageDataGenerator(rescale=1./255)
batch_size=64

train_generator = train_datagen.flow_from_dataframe(
                                                    dataframe=train_df,
                                                    directory=training_set_path,
                                                    x_col="Image",
                                                    y_col="Class",
                                                    subset="training",
                                                    target_size=(64,64),
                                                    batch_size=64,
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode='categorical')

valid_generator=train_datagen.flow_from_dataframe(
                                                    dataframe=train_df,
                                                    directory=training_set_path,
                                                    x_col="Image",
                                                    y_col="Class",
                                                    subset="validation",
                                                    batch_size=64,
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode="categorical",
                                                    target_size=(64,64))

test_generator = test_datagen.flow_from_dataframe(
                                                        dataframe=test_df,
                                                        directory=test_set_path,
                                                        x_col="Image",
                                                        y_col=None,
                                                        seed=42,
                                                        shuffle=False,
                                                        target_size=(64,64),
                                                        batch_size=64,
                                                        class_mode=None)
'''
model.fit_generator(
                    train_generator,
                    steps_per_epoch=2000,
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=800)
'''
# Create a loss history
history = LossHistory()


#Fitting the model
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
classifier.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)
print(STEP_SIZE_TRAIN,STEP_SIZE_VALID,STEP_SIZE_TEST)
# Save loss history to file
loss_history_path = os.path.join(script_dir, '../loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()

#Evaluation the model
classifier.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_TEST)

#Predicting the output
test_generator.reset()
pred=classifier.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

predicted_class_indices=numpy.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
filenames=filenames[:3200]
results=pandas.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results12.csv",index=False)


import numpy as np
from keras.preprocessing import image
test_image=image.load_img(r'E:\course\hellow ml !\data set\9d34462453e311ea\dataset\Test_images\image119.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)