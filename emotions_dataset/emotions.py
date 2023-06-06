import os 
import shutil
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import pickle


key = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happy', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}
def make_class_directory(name: str, total_count: int):
    """
    The function creates a directory for each class of images and copies a specified number of images
    into each directory.
    
    :param name: The name of the class or category for which the directory is being created
    :type name: str
    :param total_count: The total number of images to be copied for each class directory
    :type total_count: int
    """

    root_path = os.path.join('emotions_dataset/images', name)
    count = 0
    for i in range(7):
        path = os.path.join(root_path, str(i))
        if not os.path.exists(path):
            os.mkdir(path)

        all_dir = os.listdir(root_path)
        # all_dir.sort(key= lambda x : x.split("_")[0])

        for image_file_name in all_dir:
            if image_file_name.endswith(f'{i}.jpg'):
                shutil.copy(os.path.join(root_path, image_file_name), path)
                count += 1
            if count == total_count:
                count = 0
                break


def make_features(name : str, image_count: int):
    """
    The function "make_features" creates histograms of images in a given directory and returns the
    features and labels as numpy arrays.
    
    :param name: The name of the class or category of images being processed
    :type name: str
    :param image_count: The parameter `image_count` is an integer that represents the number of images
    to be used for each class in the dataset
    :type image_count: int
    :return: The function `make_features` returns two numpy arrays: `features` and `labels`. The
    `features` array contains the histogram features of the images in the dataset, and the `labels`
    array contains the corresponding labels for each image.
    """


    make_class_directory(name, image_count)
    make_class_directory(name, image_count)
    make_class_directory(name, image_count)

    features = []
    labels = []

    root_path = os.path.join('emotions_dataset/images', name)
    for class_folder in range(7):
        path = os.path.join(root_path, str(class_folder))
        
        all_dir = os.listdir(path)

        for image_file in all_dir:
            image_path = os.path.join(path, image_file)
            image = cv2.imread(image_path)
            hist = cv2.calcHist(image, [0], None, [256], [0, 256])
            labels.append(class_folder)
            hist_arr = np.array(hist)
            features.append(hist_arr)
    
    return np.array(features), np.array(labels)

def make_text_document(name: str, features: list, labels: list):
    """
    The function creates two text files containing the features and labels of an emotions dataset.
    
    :param name: The name of the text document that will be created. It will be used as a prefix for the
    file names of the features and labels text files
    :type name: str
    :param features: The "features" parameter is a list of features or attributes associated with an
    image or dataset. In this case, it seems to be a list of lists, where each inner list represents the
    features of a single image
    :type features: list
    :param labels: The `labels` parameter is a list of labels corresponding to the features in the
    `features` parameter. Each label represents the emotion or expression depicted in the corresponding
    feature image
    :type labels: list
    """


    with open(f"emotions_dataset/images/{name}_features.txt", "w") as file:
        for feature in features:
        #     file.write(f"{list(feature)}\n")
            file.write(f"{list(map(list, feature))}\n")

    with open(f"emotions_dataset/images/{name}_labels.txt", "w") as file:
        for label in labels:
            file.write(f"{label}\n")



def svm_model(train_features, train_labels):
    """
    This function trains a support vector machine (SVM) model on a set of features and labels, and saves
    the trained model using pickle.
    
    :param train_features: The training features are the input data used to train the SVM model. These
    features can be any type of data that can be represented numerically, such as images, audio signals,
    or text data
    :param train_labels: The `train_labels` parameter is a list or array containing the labels or target
    values corresponding to the `train_features` parameter. In other words, it contains the correct
    output values for each input feature vector in the training set. These labels are used to train the
    SVM model to predict the correct label
    """
    nsamples, nx, ny = train_features.shape
    train_dataset = train_features.reshape((nsamples, nx*ny))
    svm_model = SVC()
    svm_model.fit(train_dataset, train_labels)

    pickle.dump(svm_model, open('emotions_dataset/emotions_model.pkl', 'wb'))



def model_predictions(test_features, test_labels):
    """
    This function loads a trained machine learning model, uses it to make predictions on test data, and
    outputs the accuracy, confusion matrix, and classification report of the predictions.
    
    :param test_features: The input features used to test the model's predictions
    :param test_labels: The true labels for the test dataset
    """

    model = pickle.load(open('emotions_dataset/emotions_model.pkl', 'rb'))

    nsamples, nx, ny = test_features.shape
    test_dataset = test_features.reshape((nsamples, nx*ny))

    test_predictions = model.predict(test_dataset)


    # Model Accuracy
    acc = metrics.accuracy_score(test_labels, test_predictions)

    print("Accuracy: ", acc)

    conf_matrix = metrics.confusion_matrix(test_labels, test_predictions)

    print('Confusion Matrix: ',conf_matrix)

    class_report = metrics.classification_report(test_labels, test_predictions)

    print('Classification Report: ', class_report)




train_features, train_labels = make_features('Training', 50)
public_test_features, public_test_labels = make_features('PublicTest', 20)
private_test_features, private_test_labels = make_features('PrivateTest', 20)

make_text_document('Training', train_features, train_labels)
make_text_document('PublicTest', public_test_features, public_test_labels)
make_text_document('PrivateTest', private_test_features, private_test_labels)

# svm_model(train_features, train_labels)

model_predictions(private_test_features, private_test_labels)