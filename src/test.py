# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout


from keras_preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (320, 240, 3)))
classifier.add(Activation('relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64,(3,3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))
classifier.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

batch_size = 16

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(320, 240),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(320, 240),
        batch_size=batch_size,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
classifier.save('first_try.h5')


