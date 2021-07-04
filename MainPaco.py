# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 0,1,2
import warnings

import TinySleepNet
import wandb

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import Notebooks.DatasetsPaco

wandb.init()

# BATCHING
# BATCH_SIZE = 4
# x = np.random.sample((100, 2))
# make a dataset from a numpy array, ma io non la ho e non la voglio
# dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)
# iter = dataset.make_one_shot_iterator()
# el = iter.get_next()

binary_labels = True
channels = ['F4']
batch_size = 64  # che per ora non ha effetto nel get item

# train data
train_dataset = Notebooks.DatasetsPaco.EDFData_TF_old("../Data/PSG1.edf", batch_size=batch_size, channels=channels,
                                                      binary_labels=binary_labels)
labels_b = []
for a, b in train_dataset:
    labels_b.extend(b)

X_binary_std = []
for X, Y in train_dataset:
    X_binary_std.append(X.numpy())
X_binary_std = np.vstack(X_binary_std)

tftraindatasetmemo = tf.data.Dataset.from_tensor_slices(
    (tf.convert_to_tensor(X_binary_std), tf.convert_to_tensor(labels_b))).batch(32).shuffle(64)

# test data
test_dataset = Notebooks.DatasetsPaco.EDFData_TF_old("../Data/PSG3.edf", batch_size=batch_size, channels=channels,
                                                     binary_labels=binary_labels)
labels_test = []
for a, b in test_dataset:
    labels_test.extend(b)

X_binary_std_test = []
for X, Y in test_dataset:
    X_binary_std_test.append(X.numpy())
X_binary_std_test = np.vstack(X_binary_std_test)

tftestdatasetmemo = tf.data.Dataset.from_tensor_slices(
    (tf.convert_to_tensor(X_binary_std_test), tf.convert_to_tensor(labels_test))).batch(32).shuffle(64)

# model train and validation
sr = int(train_dataset.sampling_rate)
num_epochs = 10
if binary_labels:
    classes = 2
else:
    classes = 5

model = TinySleepNet.TinySleepNet(sr, len(channels), classes)

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(num_epochs):

    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in tftraindatasetmemo:
        with tf.GradientTape() as tape:
            y_ = model(x)
            loss = loss_fn(y, y_)
        grads = tape.gradient(loss, model.trainable_variables)
        # wandb.log({'gradients': grads})
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss)  # loss = loss.numpy()
        epoch_accuracy.update_state(y, y_)

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    wandb.log({'epoch_loss': epoch_loss_avg.result().numpy(), 'epoch_accuracy': epoch_accuracy.result().numpy()})

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

# print('train_loss_result {}, train_accuracy_result {}'.format(train_loss_results, train_accuracy_results))
#
# model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
# history = model.fit(tftraindatasetmemo, epochs=num_epochs, batch_size=32) # con dataset o con tfdataset, che cosa cambia? # history = model.fit(X_binary_std, np.array(labels_b), epochs=num_epochs), callbacks=WandbCallback()
# # model.summary()
# plt.plot(history.history['acc'])
# plt.plot(history.history['loss'])
# plt.savefig('train')

test_accuracy = tf.keras.metrics.Accuracy()
for x, y in tftestdatasetmemo:
    logits = model(x)
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(predictions, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))