import tensorflow as tf
import os
import skimage
from skimage import data
from skimage import transform 
from skimage.color import rgb2gray
import matplotlib.pyplot as plt 
import numpy as np
import random

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = ""
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)
test_images, test_labels = load_data(test_data_directory)
# Count the number of labels
# print(len(set(labels)))

# Make a histogram with 62 bins of the `labels` data
# plt.hist(labels, 62)

# Show the plot
# plt.show()


# Get the unique labels 
unique_labels = set(labels)

# Initialize the figure
# plt.figure(figsize=(15, 15))

# --------------------preprocessing-------------------------------
# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

images28 = rgb2gray(np.array(images28))
test_images28 = rgb2gray(np.array(test_images28))
# # Set a counter
# i = 1

# # For each unique label,
# for label in unique_labels:
#     # You pick the first image for each label
#     image = images28[labels.index(label)]
#     # Define 64 subplots 
#     plt.subplot(8, 8, i)
#     # Don't include axes
#     plt.axis('off')
#     # Add a title to each subplot 
#     plt.title("Label {0} ({1})".format(label, labels.count(label)))
#     # Add 1 to the counter
#     i += 1
#     # And you plot this first image 
#     plt.imshow(image, cmap="gray")
    
# # Show the plot
# plt.show()


# --------------------------------Modeling ----------------------------
# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3001):
        _, loss_value = sess.run([train_op, loss], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss_value)

    # --------------------------------Evaluating ----------------------------
    # Pick 10 random images
    # sample_indexes = random.sample(range(len(images28)), 10)
    # sample_images = [images28[i] for i in sample_indexes]
    # sample_labels = [labels[i] for i in sample_indexes]

    # Run the "correct_pred" operation
    predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

    # Calculate correct matches 
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])                        
    
    # Calculate the accuracy
    accuracy = match_count / len(test_labels)

    # Run the "correct_pred" operation
    o_predicted = sess.run([correct_pred], feed_dict={x: images28})[0]

    # Calculate correct matches 
    o_match_count = sum([int(y == y_) for y, y_ in zip(labels, o_predicted)])                        
    
    # Calculate the accuracy
    o_accuracy = o_match_count / len(labels)

    # Print the accuracy
    print("Accuracy for Train Data: {:.3f}".format(o_accuracy))
    print("Accuracy for Test Data: {:.3f}".format(accuracy))

    # Print the real and predicted labels
    # print(test_labels)
    # print(predicted)

    # Display the predictions and the ground truth visually.
    # fig = plt.figure(figsize=(10, 10))
    # for i in range(len(test_images28)):
    #     truth = test_labels[i]
    #     prediction = predicted[i]
    #     plt.subplot(5, 2, 1+i)
    #     plt.axis('off')
    #     color='green' if truth == prediction else 'red'
    #     plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
    #             fontsize=12, color=color)
    #     plt.imshow(test_images28[i],  cmap="gray")

    # plt.show()