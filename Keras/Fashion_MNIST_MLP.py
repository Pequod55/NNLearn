import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
#import tqdm
#import tqdm.auto

def normalize(images, labels):
    images=tf.cast(images, tf.float32)
    images/=255
    return images, labels

#tqdm.tqdm=tqdm.auto.tqdm
tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.ERROR)

dataset, metadata=tfds.load('fashion_mnist', as_supervised=True, with_info=True)
trainset, testset=dataset['train'], dataset['test']
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


num_train_examples=metadata.splits['train'].num_examples
num_test_examples=metadata.splits['test'].num_examples

"""
print(num_train_examples)
print(num_test_examples)
"""
trainset=trainset.map(normalize)
testset=testset.map(normalize)
"""
plt.figure(figsize=(10,10))
x=0
for i,j in trainset.take(25):
    i=i.numpy().reshape((28,28))
    plt.subplot(5,5,x+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(i, cmap=plt.cm.binary)
    plt.xlabel(class_names[j])
    x+=1
plt.show()
"""

model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
BATCH_SIZE=30
trainset=trainset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
testset=testset.batch(BATCH_SIZE)

history=model.fit(trainset, epochs=10, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
right=0
current=1
for img , lbl in testset.take(10):
    img=img.numpy()
    lbl=np.array(lbl)
for i in range(10):
    cat=class_names[lbl[i]]
    image=img[i]
    prediction=class_names[np.argmax(model.predict(np.array([image])))]
    if  prediction==cat:
        right+=1
    Accuracy=right/current*100
    plt.imshow(image.reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel("Given = {0}, Prediction= {1}, Current Accuracy = {2}".format(cat, str(prediction), Accuracy))
    plt.show()
    current+=1

print(Accuracy,"%")
model.save("C:\\Users\\yashm\\Documents\\TensorflowLearn\\Saved_Models\\Fashion MNIST MLP.h5")
