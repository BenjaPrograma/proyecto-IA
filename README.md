# Proyecto Topicos Avanzados de IA
- Repositorio utilizado/ejecutado en COLAB: 
https://colab.research.google.com/drive/1I-_81IKBNIwOXX6rJqAllNHTRptX1oun?authuser=1#scrollTo=0AJVw3Fp1AdO


### ENTRENAR BASELINE
!python r2r_src/train.py 


¿Que es el feature predictor? (TEMPORAL DIFFERENCE TASK)






### HIPOTESIS

La parte de "Saber si la ruta matchea con la instrucción" es basicamente esto

Como funciona es, corre el programa hasta antes que termine el stop. Y le hace la pregunta si la ruta que vio hace match con la instrucción, la cual se cambia con prob 0.5 con otra del batch. 
Por lo que esta tarea se vuelve sencilla con el tiempo. (COMPROBAR ESTO viendo la LOSS PROGRESSION DE LA TASK) 

-El resultado que tienen es malo en VAL SEEN y bueno en VAL UNSEEN, esto indica según los autores que funciona bien como un regularizador, y mejora la generalizacion


Mejor tarea es, de la ruta que hace hasta que termine el stop. Preguntarle si la ruta que vio es la de la instrucción, donde el cambio en la instruccion se da con una chance de, 0.5 pero sobre un set de cosas que pueden ocurrir (al menos una de ellas) que cambiarian solo UN TROZO de toda la instrucción, por ejemplo, cambiar un objeto por otro que no esté en la ruta.
finalmente cambiar orden de objetos y que detecte si es falsa.

-Probar si esta tarea es posible sin tener que usar un detector de imagenes, es decir si entrena con lo dicho anteriormente, quiza probar cambiando todos los objetos en un inicio, e ir aumentando la probabilidad de cambiar más objetos con el tiempo (a medida que va aprendiendo) llegando a cambiar solo un objeto y que sepa que es una ruta falsa
-Si hay que meter el detector de objetos hacerlo.