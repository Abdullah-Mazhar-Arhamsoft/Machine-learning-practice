import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt


key = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happy', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}

def predict_emotions(image: str):
    """
    This function takes in an image directory, loads a pre-trained model, predicts the emotion of each
    image in the directory, and displays the image with its predicted emotion.
    
    :param image: The parameter "image" is a string that represents the path to a directory containing
    images for which we want to predict emotions
    :type image: str
    """

    all_dir = os.listdir(image)

    model = pickle.load(open('emotions_dataset/emotions_model.pkl', 'rb'))

    for path in all_dir:
        image_path = os.path.join(image, path)
        img = cv2.imread(image_path)
        hist = cv2.calcHist(img, [0], None, [256], [0, 256])
        # print(hist.shape)
        list_feature = [hist]
        list_feature = np.array(list_feature)
        # print(list_feature.shape)

        nsamples, nx, ny = list_feature.shape
        test_dataset = list_feature.reshape((nsamples, nx*ny))
        # print(test_dataset.shape)

        test_prediction = model.predict(test_dataset)

        emotion = key[test_prediction[0]]

        display_image(img, emotion)


def display_image(img, emotion):
    """
    This function displays an image with a text label indicating the emotion associated with the image.
    
    :param img: The image that you want to display
    :param emotion: The emotion parameter is a string that represents the emotion being displayed in the
    image. It is used to add a text label to the image
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.text(21, 1, emotion, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.8))
    plt.axis('off')
    plt.show()


image_path = 'emotions_dataset/images/Test'
img, emotion = predict_emotions(image_path)


