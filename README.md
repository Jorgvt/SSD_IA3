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
- [ ] Añadir opciones de normalización a los datasets custom.
- [ ] Hay que ver cómo hacemos para asegurarnos de que todos los datasets tienen el mismo mapeado de labels
- [ ] Normalizar datos con los valores de otros datos, para poder hacerlo bien entre test y train y todo eso. Ahora mismo se están normalizando con sus propios datos, pero meh.

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
| Normalización datasets custom | | | :construction_worker: |

:construction_worker: &rarr; Work in progress
:white_check_mark: &rarr; Completed