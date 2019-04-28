import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt


def normalize(images, labels):
    images=tf.cast(images, tf.float32)
    images/=255
    return images, labels

tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.ERROR)

dataset, metadata=tfds.load('fashion_mnist', as_supervised=True, with_info=True)
trainset, testset=dataset['train'], dataset['test']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


num_train_examples=metadata.splits['train'].num_examples
num_test_examples=metadata.splits['test'].num_examples


trainset=trainset.map(normalize)
testset=testset.map(normalize)
testing=testset

#with tf.Session() as sess:
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(12, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
BATCH_SIZE=30
trainset=trainset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
testset=testset.batch(BATCH_SIZE)
testing=testing.batch(1000)
history=model.fit(trainset, epochs=1, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
right=0
current=1
for img , lbl in testing.take(1000):
    img=img.numpy()
    lbl=np.array(lbl)
for i in range(1000):
    cat=class_names[lbl[i]]
    image=img[i]
    prediction=class_names[np.argmax(model.predict(np.array([image])))]
    if  prediction==cat:
        right+=1
    Accuracy=right/current*100
    #plt.imshow(image.reshape(28,28), cmap=plt.cm.binary)
    #plt.xlabel("Given = {0}, Prediction= {1}, Current Accuracy = {2}".format(cat, str(prediction), Accuracy))
    #plt.show()
    current+=1

print(Accuracy,"%")
model.save("C:\\Users\\yashm\\Documents\\TensorflowLearn\\Saved_Models\\Fashion MNIST CNN.h5")
