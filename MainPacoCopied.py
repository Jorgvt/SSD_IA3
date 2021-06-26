import matplotlib.pyplot as plt
import tensorflow as tf

import Notebooks.DatasetsPaco
import TinySleepNet

if __name__ == '__main__':

    binary_labels = True
    channels = ['F4']
    dataset = Notebooks.DatasetsPaco.EDFData_TF_old("../Data/PSG1.edf", batch_size=64, channels=channels, binary_labels=binary_labels)
    # dataset = Notebooks.DatasetsPaco.EDFData("../Data/PSG1.edf", channels=['F4', 'O2'], binary_labels=True)

    sr = int(dataset.sampling_rate)
    num_epochs = 10
    classes = 5
    if binary_labels:
        classes = 2
    model = TinySleepNet.TinySleepNet(sampling_rate=sr, channels=len(channels), classes=classes)

    # method 1
    # optimizer = tf.keras.optimizers.Adam(1e-4)
    # cce = tf.keras.losses.SparseCategoricalCrossentropy()

    # loss_epoch = []
    # for epoch in range(num_epochs):
    #
    #     loss_samples = []
    #     for x, y in dataset:
    #         with tf.GradientTape() as tape:
    #             # x = tf.keras.backend.constant(x)
    #             preds = model(x)
    #             # y = tf.keras.backend.constant(y)
    #             loss = cce(y, preds)
    #             loss_samples.append(loss.numpy())
    #         grads = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     loss_samples_meaned = mean(loss_samples)
    #     loss_epoch.append(loss_samples_meaned)

    # method 2
    # def loss(model, x, y, training):
    #     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    #     # training=training is needed only if there are layers with different
    #     # behavior during training versus inference (e.g. Dropout).
    #     y_ = model(x, training=training) # es como si lo prediga a lazar? preds = model(x) equiprobable siempre
    #     return loss_object(y_true=y, y_pred=y_)
    #
    # def grad(model, inputs, targets):
    #     with tf.GradientTape() as tape:
    #         loss_value = loss(model, inputs, targets, training=True)
    #     return loss_value, tape.gradient(loss_value, model.trainable_variables)
    #
    # # Keep results for plotting
    # train_loss_results = []
    # train_accuracy_results = []
    #
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    #
    # for epoch in range(num_epochs):
    #     epoch_loss_avg = tf.keras.metrics.Mean()
    #     epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    #
    #     # Training loop - using batches of 32?
    #     for x, y in dataset: # train_dataset
    #         # Optimize the model
    #         loss_value, grads = grad(model, x, y)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #
    #         # Track progress
    #         epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    #         # Compare predicted label to actual label
    #         # training=True is needed, only if there are layers with different
    #         # behavior during training versus inference (e.g. Dropout).
    #         epoch_accuracy.update_state(y, model(x, training=True))
    #         # break # veo siempre el mismo batch
    #
    #     # End epoch
    #
    #     train_loss_results.append(epoch_loss_avg.result())
    #     train_accuracy_results.append(epoch_accuracy.result())
    #
    #     # if epoch % 50 == 0:
    #     print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
    #                                                                 epoch_loss_avg.result(),
    #                                                                 epoch_accuracy.result()))
    #
    # fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    # fig.suptitle('Training Metrics')
    #
    # axes[0].set_ylabel("Loss", fontsize=14)
    # axes[0].plot(train_loss_results)
    #
    # axes[1].set_ylabel("Accuracy", fontsize=14)
    # axes[1].set_xlabel("Epoch", fontsize=14)
    # axes[1].plot(train_accuracy_results) # perche son sempre gli stessi samples
    # plt.savefig("mygraph.png") # plt.show()

    # method # 3

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]) # pero con # self.classes-1, sigmoid, si binarycrossentropy
    model.fit(dataset, epochs=num_epochs)
    model.summary()