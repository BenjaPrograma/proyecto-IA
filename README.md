# Proyecto Topicos Avanzados de IA
- Repositorio utilizado/ejecutado en COLAB: 
https://colab.research.google.com/drive/1I-_81IKBNIwOXX6rJqAllNHTRptX1oun?authuser=1#scrollTo=0AJVw3Fp1AdO

train.py -> train_val -> (R2RBatch -> load_datasets), train -> Seq2Seq init -> Seq2Seq.train


- train.py -> train_val() 
corre los splits.
Carga las features de imagenes
Llama a R2RBatch *
LLama a train() (con R2RBatch = train_env)

- R2RBatch (en env.py)
carga las features de imagenes
carga en self.DATA las instrucciones (con load_datasets()) en un dict, que apendea a DATA, se usa un tokenizer para el primer encoding (para pasarlo a la lstm) ***
y la info para correr el simulador
CREO QUE LA GRACIA ES Q _get_obs tiene de todo.
y todo esta cargado en R2RBatch (que despues se llama train_env)

- load_datasets esta en utils.py y lee los json *

- train() (de train.py) 

Aca estan los logs,
se llama al model learner con el train_env
se carga speaker
se llama al SEQ2SEQ.train()

-SEQ2SEQ

### ENTRENAR BASELINE
!python r2r_src/train.py 


¿Que es el feature predictor? (TEMPORAL DIFFERENCE TASK)

nvidia-docker run -it --mount type=bind,source="/mnt/d/BACKUP 05-10-2021 FORMAT/Lifestein/UC/UC 2021-2/Topicos de IA/Proyecto/proyecto-IA/data/matterport_unzip",target=/root/mount/proyecto-IA/data/matterport_unzip --volume `pwd`:/root/mount/proyecto-IA mattersim:9.2-devel-ubuntu18.04

export MATTERPORT_DATA_DIR=

## TODO
-- Ver como cambiar OBJS por NADA
-- Ver como cambiar OBJS por OBJS no presentes
-- Swappear OBJS dentro de una oracion

## COSAS WORK

- COMO OBTENER OBJETOS DE UN PATH_ID?
Se pueden obtener de las dense_features y del matterport3dsimulator que nos provee
el repositorio de nuestros amigos https://github.com/cacosandon/360-visualization










## DETALLE DE PONER EN TRAINVAL_VOCAB LOS OBJETOS QUE SAQUE DEL VOCAB DE LOS SCANS DE AHI
FIXED PERO EL VOCAB_VAL ES IGUAL AL VOCAB_TRAIN, PERO LOS UNSEEN SON UNSEEEN.


+ Aumentar el vocabulario con los objetos nuevos

le pasaremos h1 + fake_ctx a nuestra red FC
fake_ctx = self.encoding(fake_seq, seq_length)

Tendremos que crear un fake_instructions que detecte objetos, y los cambie.
Para eso crear un nuevo .json con las siguientes columnas
objetos.json = todos los que son objetos del vocab
r2r_train_objinfo.json = con nuevas keys de objetos presentes 



- Se hara en 12k Iterations, ya que después sigue subiendo el  val_seen y train, pero el unseen muy muy poco. (0.2 puntos en 10k de iteraciones) El paper usa 80k
pero son 360 minutos en 22k en colab pro.

- Se hara con 
--denseObj --name XXX

### FILES

#### base
- nlp.py genera los obj spacy en data
- instruction_mod.py genera los obj densefeatures en data
- script en repositorio cacosandon genera los obj matt3d en data.
- scans_unseen.py genera "scans_val_unseen_test_set.pkl" un set pickleado para no leer aquellos scans al formar el resto de los datos.

#### compuestos
(generado por obj_vocab_maker.py)
- OBJS_CERTAIN_247.TXT/.pkl: Tiene la triple interseccion de densefeat/matt3d y spacy de las instrucciones, de todos los objetos en común, por lo que es lo más probable que si sean objetos reales.

- scanid_object_maker.py -> Genera "scanid_to_objs_and_aux_objs.pkl"

- "scanid_to_objs_and_aux_objs.pkl" scanid, con 2 listas, una con los objetos por las instrucciones (con spacy), y otro con los objetos aux.



#### Funcion reemplazo objetos en un scanid por falsos

+ Cuando se lean los R2R train y val_seen? o creo que solo R2R Train, hay que leer el doc
+ Agregar el vocab nuevo a train_vocab.txt y a trainval_vocab.txt?
+ Vamos a hacer que si en una instrucción hay 2 objetos o más seguidos, se consideran como que hay que reemplazar todos esos. Siempre reemplazar por palabras que esten en el objs_certain, pero que no estén ni en la instrucción_obj ni en el aux_obj
+ Vamos a generar una funcion que con prob=X cambie los objetos que detecte (si estan seguidos cuentan como 1), esto tiene que ser antes del tokenizer, y luego tenemos que pasarselo al tokenizer, y luego al encoding y ver que es eso de Variable.require_grad=False, para asegurar que no quede con descenso de gradiente.
+ Finalmente en la función vemos si le pasamos el original o el falso y que prediga el label, concatenando el h1 + fake_ctx


### BASELINES:
RIAL_mat_dense_baseline_0_85 -> 0.4 unseen
- Mat no entrena (loss)
- Rendimiento = con Dense Features

MG-AuxRN
- Mat no entrena, el resto sí, lentamente, podría ser mejor que 0.4 unseen

### HIPOTESIS

La parte de "Saber si la ruta matchea con la instrucción" es basicamente esto

Como funciona es, corre el programa hasta antes que termine el stop. Y le hace la pregunta si la ruta que vio hace match con la instrucción, la cual se cambia con prob 0.5 con otra del batch. 
Por lo que esta tarea se vuelve sencilla con el tiempo. (COMPROBAR ESTO viendo la LOSS PROGRESSION DE LA TASK) 

-El resultado que tienen es malo en VAL SEEN y bueno en VAL UNSEEN, esto indica según los autores que funciona bien como un regularizador, y mejora la generalizacion


Mejor tarea es, de la ruta que hace hasta que termine el stop. Preguntarle si la ruta que vio es la de la instrucción, donde el cambio en la instruccion se da con una chance de, 0.5 pero sobre un set de cosas que pueden ocurrir (al menos una de ellas) que cambiarian solo UN TROZO de toda la instrucción, por ejemplo, cambiar un objeto por otro que no esté en la ruta.
finalmente cambiar orden de objetos y que detecte si es falsa.

-Probar si esta tarea es posible sin tener que usar un detector de imagenes, es decir si entrena con lo dicho anteriormente, quiza probar cambiando todos los objetos en un inicio, e ir aumentando la probabilidad de cambiar más objetos con el tiempo (a medida que va aprendiendo) llegando a cambiar solo un objeto y que sepa que es una ruta falsa
-Si hay que meter el detector de objetos hacerlo.