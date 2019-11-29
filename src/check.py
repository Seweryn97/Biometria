import cv2
import keras


CATEGORIES= ["32","33"]

def check(filepath):
    IMG_SIZE = 70
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)


model= keras.models.load_model('first_try.h5')
prediction = model.predict([check('1_32.png')])
print(prediction)

