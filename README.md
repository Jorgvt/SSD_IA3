# SSD_IA3
Sleep Stage Detection study as part of the final thesis of the Master IA^3.

# Cosas que hacer
- [x] Conseguir un modelo que sea capaz de aprender algo.
- [ ] Probar con todos los datos.
- [ ] Plots que muestren las labels y las predicciones en función del tiempo.
![Label_vs_Time](Images/label_vs_time_sample.png)
- [ ] Matriz de transiciones entre estados.
- [ ] Probar un modelo de clasificación binaria que solo tenga que distinguir entre despierto y durmiendo.
- [ ] **Crear una pipeline donde solamente tengamos que cambiar el modelo y tocar algunas palancas.**
- [ ] **Incluir W&B en la pipeline anterior**. Esto implica decidir todas las cosas que nos gustaría trackear.

## Cosas que podría estar bien trackear en todos los experimentos
- Arquitectura
- Canales usados
- Matriz de confusión
- Plot de labels/preds en función del tiempo
- Accuracy
- ¿F1?

# Distribución de las tareas
| Tarea | Pablo | Francesco | Jorge |
|-------|:-------:|:-----------:|:-------:|
| Problema binario | | :construction_worker: | |
| Todos los datos | :construction_worker: | | |
| Visualizaciones | | | :construction_worker: |

:construction_worker: &rarr; Work in progress
:white_check_mark: &rarr; Completed

Notes:
- alternativamente a un for con un if else?
- a sleep stage W le associo il label 1, pero tendria que apuntarlo? No bien
- tensorflow.python.framework.errors_impl.InvalidArgumentError:  Conv2DCustomBackpropInputOp only supports NHWC.
     [[node gradient_tape/sequential/conv1d_3/conv1d/Conv2DBackpropInput
- self.data = mne.io.read_raw_edf(path) # self.sampling_frequency = int(self.data.info['sfreq']) como mejoria
- limpiar el main in blocchi, anche concettuali
- estender a tutti i pazienti il train = non posso piu np.transpose(np.expand_dims(data, 1), (0, 2, 1)) 
- https://www.tensorflow.org/guide/keras/functional
- https://keras.io/api/layers/convolution_layers/convolution1d/
- https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
- 2021-06-23 19:31:08.433779: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
  2021-06-23 19:31:08.433797: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
  2021-06-23 19:31:08.433812: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (ubuntu): /proc/driver/nvidia/version does not exist
  2021-06-23 19:31:08.433953: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
  To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
- https://wandb.ai/jorgvt/test-pth?workspace=user-jorgvt
- y_ = preds = 64x5, y = 64,1