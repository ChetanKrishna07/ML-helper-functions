import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras import layers
import os
import datetime

def plot_decision_boundary(model, X, y):
  """
  Plots the decision boundary created by a model predicting X
  """

  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = X[:, 0].min() - 0.1, X[:,0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:,1].max() + 0.1

  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))

  # Create X values
  x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2d arrays together

  # Make predictions
  y_pred = model.predict(x_in, verbose=0)

  # Check for multi-class
  if len(y_pred[0]) > 1:
    print("Doing multiclass classification")
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("Doing binary classification")
    y_pred = np.round(y_pred).reshape(xx.shape)

  # Plot
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  
  
import itertools
from sklearn.metrics import confusion_matrix

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(7,7), text_size=15):

  # Creating the confusion matrix

  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  n_classes = cm.shape[0]

  # Lets prettify it
  fig, ax = plt.subplots(figsize=figsize)

  # creating matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  # set labels to be classes
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Labeling the axis
  ax.set(title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,)

  # set x-axis labels to bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Adjust label size
  ax.yaxis.label.set_size(text_size)
  ax.xaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  threshold = (cm.max() + cm.min()) / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else  "black",
            size=text_size)


def learning_rate_analysis(history):
  lrs = history.history["lr"]
  plt.semilogx(lrs, history.history["loss"])
  plt.xlabel("Learning  rate")
  plt.ylabel("Loss")
  plt.title("Finding the ideal learning rate")
  plt.show()
  
import matplotlib.pyplot as plt

def plot_image(X, y, idx, classes):
  """
  plots an image with its label at a given index

  args:
    X: images
    y: labels
    idx: index of the image to be plotted
    classes: list of classes
  """
  plt.imshow(X[idx], cmap=plt.cm.binary)
  plt.title(classes[y[idx]])
  plt.axis(False)
  
# Plot multiple random images
def plot_random(X, y, classes):

  """
  Plots 8 random images with its label
  args:
    X: images
    y: labels
    classes: list of classes
  """
  import random
  plt.figure(figsize=(10, 7))

  for i in range(8):
    ax = plt.subplot(2, 4, i + 1)
    plot_image(X, y, random.choice(range(len(X))), classes)


def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from file and turns it into a tensor and reshapes it
  """

  # read the image
  img = tf.io.read_file(filename)

  # decode into an tensor
  img = tf.image.decode_image(img)

  # re-size the image
  img = tf.image.resize(img, size=[img_shape, img_shape])

  # rescale the image
  img = img/255.

  return img


def pred_and_plot(model, filename, class_name):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """

  img = load_and_prep_image(filename=filename)
  pred = model.predict(tf.expand_dims(img, axis=0), verbose=0)
  if len(pred[0] > 1):
    pred_idx = tf.argmax(pred[0])
    pred_class = class_names[pred_idx]
    conf = pred[0][pred_idx] * 100
  else:
    pred_class = class_name[int(tf.round(pred))]
    conf = pred[0][0] * 100 if pred[0][0] > 0.5 else(1 - pred[0][0]) * 100

  plt.figure(figsize=(4, 4))
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}. Confidence: {conf}%")
  plt.axis("off")
  
  

def plot_loss_curves(history):

  """
  Plots the history curve seperating valaidation and training loss and accuracy

  history: history object from model.fit

  """

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(loss))

  # Plot loss
  plt.figure()
  plt.plot(epochs, loss, label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  plt.figure()
  #Plot accuracy
  plt.plot(epochs, accuracy, label="training_accuracy")
  plt.plot(epochs, val_accuracy, label="val_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()

def create_transfer_model(model_url, num_classes=10):
  """
  Takes a TensorFlow Hub URL and create a Keras Sequential Model

  Args:
    model_url (str) : A TensorFlow Hub feature extraction URL
    num_classes (int) : Number of output neurons in the output layer,
                        should be equal to the number of target classes

  Returns:
    model (tf.keras.Model) : An uncompiled keras Sequential model instance in Keras Functional API
  """

  feature_extraction_layer = hub.KerasLayer(model_url,
                                            trainable=False, # These model will be retrained, but freeze these layers which already learned the patters
                                            name="feature_extraction_layer",
                                            input_shape=IMG_SHAPE+(3,))

  model = tf.keras.Sequential([
      feature_extraction_layer,
      layers.Dense(num_classes, activation="softmax", name="output_layer")
  ])

  return model



def walk_through(root):
    """
    Walks through each directory and subdirectory starting from root
    """
    for dirpath, dirnames, filenames in os.walk(root):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def create_tb_cb(dir_name, exp_name):
    """
    Creates a tensorboard call back at a dir_name with exp_name
    """
    log_dir = dir_name + "/" + exp_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log file to : {log_dir}")
    return tensorboard_callback
  
def get_classnames(train_dir):
    """
    Returns a list of classes in the train_dir
    """
    return np.array(sorted(os.listdir(train_dir)))
