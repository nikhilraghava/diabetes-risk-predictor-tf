import tensorflow as tf
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('data.csv', index_col = False)
# Features
X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age']]
# Label
y = df[['Outcome']]
# Split training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Convert single class to double class
df['NotDiabetes'] = 1-df['Outcome']
y = df[['Outcome', 'NotDiabetes']]
# Convert X and y to CSV
X.to_csv('features.csv', index= False, header=False)
y.to_csv('labels.csv', index= False, header=False)
# Get data from features.csv and labels from labels.csv
data = pd.read_csv('features.csv', index_col = False)
labels = pd.read_csv('labels.csv', index_col=False)
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)

# Variables
tf_x = tf.placeholder(tf.float32)
tf_y = tf.placeholder(tf.float32)

# Hidden layers
hidden_1 = 1000
hidden_2 = 1000
hidden_3 = 1000
hidden_4 = 1000
hidden_5 = 1000
hidden_6 = 1000

# Classes
n_classes = 2
# Batch size
batch_size = 100

# Model
def neural_network_model(data):
    # Weights and biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([8, hidden_1])),
                      'biases': tf.Variable(tf.random_normal([hidden_1]))}
    
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([hidden_1, hidden_2])),
                      'biases': tf.Variable(tf.random_normal([hidden_2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([hidden_2, hidden_3])),
                      'biases': tf.Variable(tf.random_normal([hidden_3]))}
    
    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([hidden_3, hidden_4])),
                      'biases': tf.Variable(tf.random_normal([hidden_4]))}
    
    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([hidden_4, hidden_5])),
                      'biases': tf.Variable(tf.random_normal([hidden_5]))}
    
    hidden_6_layer = {'weights': tf.Variable(tf.random_normal([hidden_5, hidden_6])),
                      'biases': tf.Variable(tf.random_normal([hidden_6]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([hidden_6, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)

    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.relu(l5)

    l6 = tf.add(tf.matmul(l5, hidden_6_layer['weights']), hidden_6_layer['biases'])
    l6 = tf.nn.relu(l6)

    output = tf.add(tf.matmul(l6, output_layer['weights']), output_layer['biases'])

    return output


def train_network(x):
    # Predict
    prediction = neural_network_model(x)
    # Cross-entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=prediction))
    # Optimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # Model will train for 10 cycles, feed forward + backprop
    epochs = 100

    # Session
    with tf.Session() as sess:
        # Initialize global variable
        sess.run(tf.global_variables_initializer())
        # Run cycles
        for epoch in range(epochs):
            epoch_loss = 0
            # Split and batch
            total_batch = int(len(X_train)/batch_size)
            X_batches = np.array_split(X_train, total_batch)
            Y_batches = np.array_split(y_train, total_batch)
            # Run over all batches
            for i in range(total_batch):
                epoch_x, epoch_y = X_batches[i], Y_batches[i]
                _, c = sess.run([optimizer, cost], feed_dict={tf_x: epoch_x, tf_y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochs, 'loss', epoch_loss)

        # Check prediction
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(tf_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        result = sess.run(accuracy, feed_dict={tf_x: X_test, tf_y: y_test})
        print("{0:f}%".format(result * 100))

train_network(tf_x)