from __future__ import print_function, division
from builtins import range, input
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, Add, Activation
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools

class ResNet:
  def __init__(self, x_train, y_train, x_test, y_test):
    self.x_train = x_train / 255.0
    self.x_test =  x_test / 255.0
    self.y_train, self.y_test = y_train.flatten(), y_test.flatten()
    self.k = len(np.unique(y_train)) # pandas -> set()   numpy -> unique()
    print("X_train.shape:", self.x_train.shape)
    print("y_train.shape:", self.y_train.shape)

    # number of classes
    print("number of columns:", self.k)

  def identityBlock(self, x, filter):
      shortcut = x
      
      # 1st layer
      x = Conv2D(filter,(3, 3), strides=(1, 1), padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation(activation='relu')(x)

      # 2nd layer
      x = Conv2D(filter,(3, 3), strides=(1, 1), padding='same')(x)
      x = BatchNormalization()(x)

      # addition
      x = Add()([x, shortcut])
      x = Activation(activation='relu')(x)

      return x

  def convBlock(self, x, filter):
      shortcut = Conv2D(filter,(3, 3), strides=(1, 1), padding='valid')(x)
      shortcut = BatchNormalization()(shortcut)
      
      # 1st layer
      x = Conv2D(filter,(3, 3), strides=(1, 1), padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation(activation='relu')(x)

      # 2nd layer
      x = Conv2D(filter,(3, 3), strides=(1, 1), padding='valid')(x)
      x = BatchNormalization()(x)

      # addition
      x = Add()([x, shortcut])
      x = Activation(activation='relu')(x)

      return x

  def fullConnection(self, x, i):
      x = Flatten()(x)
      x = Dropout(0.2)(x)
      x = Dense(1024, activation='relu')(x)
      x = Dropout(0.2)(x)
      x = Dense(self.k, activation='softmax')(x)

      model = Model(i, x)

      return model

  def CIFAR_Improved(self, x_train, y_train, x_test, y_test):  
      # Compile and fit
      # using the GPU for this
      model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
      # Fit
      r = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=50)

      # Fit with data augmentation
      # if you run this AFTER calling the previous model.fit(), it will CONTINUE training where it left off
      batch_size = 32
      data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
      train_generator = data_generator.flow(self.x_train, self.y_train, batch_size)
      steps_per_epoch = x_train.shape[0] // batch_size
      r = model.fit(train_generator, validation_data=(self.x_test, self.y_test), steps_per_epoch=steps_per_epoch, epochs=50)

      # Plot loss per iteration
      plt.plot(r.history['loss'], label='loss')
      plt.plot(r.history['val_loss'], label='val_loss')
      plt.legend()
      plt.show()

      # Plot accuracy per iteration
      plt.plot(r.history['accuracy'], label='acc')
      plt.plot(r.history['val_accuracy'], label='val_acc')
      plt.legend()
      plt.show()

      p_test = model.predict(self.x_test).argmax(axis=1)
      cm = confusion_matrix(self.y_test, p_test)
      self.plot_confusion_matrix(cm, list(range(10)))

      # Label mapping
      labels = '''airplane
      automobile
      bird
      cat
      deer
      dog
      frog
      horse
      ship
      truck'''.split()

      # Show some misclassified examples
      misclassified_idx = np.where(p_test != self.y_test)[0]
      i = np.random.choice(misclassified_idx)
      plt.imshow(self.x_test[i], cmap='gray')
      plt.title("True label: %s Predicted: %s" % (labels[self.y_test[i]], labels[p_test[i]]));
      plt.show()

  # plot confusion matrix
  def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):

      # this function prints and plots the confusion matrix.
      # Normalization can be applied by setting 'normalize=True'

      if normalize:
          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          print("Normalized confusion matrix")
      else:
          print('Confusion matrix, without normalization')

      print(cm)
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.title(title)
      plt.colorbar()
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)

      fmt = '.2f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment='center',
                  color='white' if cm[i, j] > thresh else 'black')

      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('predicted label')
      plt.show()
  

if __name__ == '__main__':
    # load in the data
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    ResNet = ResNet( x_train, y_train, x_test, y_test)
    
    # conv
    i = Input(shape = x_train[0].shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu')(i)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3,3), strides=(2, 2))(x)
    
    x = ResNet.identityBlock(x, filter=64)
    x = ResNet.convBlock(x, filter=64)
    model = ResNet.fullConnection(x, i)
    print(model.summary())
    # cifar
    resnet = ResNet.CIFAR_Improved(x_train, y_train, x_test, y_test)
   
    
