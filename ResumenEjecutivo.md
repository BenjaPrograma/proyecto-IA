## Funcionamiento de Tarea Auxiliar
- Importante mencionar que matching y las tareas asociadas solo se entrenan durante
teacher forcing. El rollout durante el entrenamiento se ejecuta una vez en teacher forcing y otra en "sample" que sería argmax, y luego se realiza el ajuste de los pesos, esto es lo propuesto por el paper original, aunque los pesos de cada parte del entrenamiento no se especifican y los considero como hiperparametros importantes.

### Lineas de código importantes
    # EMPEZANDO EL ROLLOUT (1 iteracion del batch)
    seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
    # _sort_batch agrupa en seq (un tensor), las instrucciones (codificadas)
    ctx, h_t, c_t = self.encoder(seq, seq_lengths)
    # se las pasa al encoder y genera el vector ctx de atencion sobre las instrucciones

    # luego se entra al episodio
    for t in range(self.episode_len):
    # se le pasa al decoder
        h_t, c_t, logit, h1 = self.decoder(..., ctx, ...)
    # al finalizar el episodio se ejecuta matching original
    # que utiliza ctx y v_ctx (atención de la instruccion y atencion visual)
    # existe otro vector vl_ctx de la atención crossmodal, el paper menciona
    # que utilizan vl_ctx, pero en el código del repo utilizan v_ctx.
    
### Lineas de código importantes para la modificacion de matching
    def gen_fake_instruction(instr):
        """ se genera una instrucción falsa seleccionando palabras al azar, tanto
        para la que se reemplaza y la selección de la palabra reemplazante
        """
        what_to_replace = random.choice("obj","dir","both")
        
        word_to_replace = random.choice([x for x in instr if x is what_to_replace])
        new_word = random.choice([x for x in list_of_what_to_replace])
        modified_instr = instr.replace(word_to_replace, new_word)
        # en realidad es el indice, y no se utiliza la función replace, ya que
        # esta reemplaza la primera ocurrencia
        return modified_instr

    def _sort_batch_fake_instruction(...):
        """ Esta función se utiliza en cada iteración para codificar una
         instrucción modificada
        """
        for ob in obs:
            fake_instr = gen_fake_instruction(...)

    # EMPEZANDO EL ROLLOUT
    # crear instrucción falsa (se hace para cada iter)
    # siguiendo los mismos pasos de la instrucción original
    seq_fake, _, seq_lengths_fake, _ = self._sort_batch_fake_instruction(obs)

    with torch.no_grad():
        ctx_fake, _, _ = self.encoder(seq_fake, seq_lengths_fake)

    # una vez terminado el episodio se ejecuta la tarea auxiliar
    # El algoritmo para matins utiliza v_ctx al igual que matching original
    # Generamos aleatoriamente el vector mix_ctx y en label almacenamos el origen
    mix_ctx = []
    label = []
    ctx = ctx[:,0,:].detach()
    ctx_fake = ctx_fake[:,0,:].detach()
    for i in range(batch_size):
        if random.random() > 0.5:
            mix_ctx.append(ctx_fake.select(0,i))
            label.append(0)
        else:
            mix_ctx.append(ctx.select(0,i))
            label.append(1)

### Lineas de código para matching episodico
Es la misma idea pero dentro de un episodio, existen varias alternativas. Según como tomamos los vectores visuales y de la instrucción, podemos variar estos según el vector del paso actual del episodio, o usando todos los vectores hasta un punto del episodio, esto para vector de instrucción y visual.

- Tomar todos los v_ctx hasta un punto y compararlos con toda la instrucción hasta un punto
- Tomar el v_ctx actual e instrucción actual asociada.
- Tomar v_ctx hasta un punto y comparar con instruccion

Actualmente estoy buscando como armar el vector de instrucción en cada punto del episodio, me confunde el funcionamiento de la función sort_batch_instruction en agent.py, puesto que pareciera que se codifica solo una porcion de la instrucción para cada batch, pero esta función se llama solo una vez y parece codificar toda la instrucción y todavia no entiendo en que parte se realiza esto o como.



## Criterios de selección de palabras
- Muchos de los pasos se precomputan, y se almacenan los datos en archivos para posterior uso durante el entrenamiento
- Archivo importante es r2r_src/nlp_spacy_nltk.py
- Se seleccionan "objetos" y "direcciones". Hay que buscar una manera de formalizar la selección de ellos dentro de las instrucciones, ya que el criterio de selección manual puede tener un sesgo.
- La precomputación es la parte más importante de formalizar, ya que con ella generamos una lista de objetos y direcciones que luego se seleccionan con la funcion gen_fake_instruction
- Luego hay que formalizar también el criterio de gen_fake_instruction, ya que con ella seleccionamos entre 1 y 2 palabras maximo (1 objeto y/o 1 direccion maximo) por trozo de instrucción. Esto con el objetivo de no cambiar mucho la instrucción, para que la tarea sea mas dificil de aprender (con la intuición (hay que formalizarla por eso) que mientras más palabras se le cambian a la instrucción original, más facil es reconocer que ha sido cambiada) 

    ### Precomputación
    - Para esto, utilizamos funciones que recorren todas las instrucciones en el set de entrenamiento y de validacion vista, para generar archivos con los objetos o direcciones.
    #### Seleccion Objetos
    - find_objs_spacy(): Usando part-of-speech tagging (POS) de Spacy (que por mis experimentos tenia mayor precision que NLTK), extraemos los sustantivos dentro de una oración, y si se encuentran sustantivos consecutivos estos se agrupan en una sola entidad. 
    - Contamos la frecuencia de ocurrencia de cada entidad en todo el set.
    - Como no todos los sustantivos son objetos, esta lista fue filtrada de manera manual por mi, para remover palabras que no eran objetos y fueron detectados como tal.


    - make_list_of_objs(threshold), genera un archivo con todos los objetos con mayor ocurrencia que threshold. 

    #### Selección direcciones y contrafactuales
    - Esta lista es generada manualmente dentro de las direcciones más comunes, falta formalizar el criterio para detectar dentro del POS (adverbios), y también como definir el contrafactual de una dirección.

## Cosas por aclarar

- Que tipo de features de lenguaje se están usando, first features only.
Entender bien los slicing que hacen, ya que en el paper original se sugiere first + last features of language.

        l_ctx = ctx[:,0,:].detach() 
        # FIRST FEATURES ONLY
        # l_ctx = torch.cat((ctx[:,0,:], ctx[:,-1,:]), dim=1).detach()
        # FIRST AND LAST

En mis experimentos utilicé solo first, el paper recomienda first + last.
Entender que significa first features v/s first + last.
Actualmente estoy corriendo experimentos con first + last.

### Update using first + last features
- The results are worse than using only the first language features, there is still the possibility that with other hyperparams the result could be better.
- What happens using first + last, is that the aux task loss converges quicker and to a lower value (0.2 v/s 0.12)
- I expected that with first + last features the results would be better, and cant find an explanation as to why it would have worse results.
- First + Last features performs better on Val_seen, both on SPL and SR with lesser iterations than with first features only. 

## Ideas por explorar
- Cambio temporal dentro de las instrucciones (esto es cambiar el orden de los trozos de instrucción) agregando más labels (la tarea dejaria de ser clasificacion binaria), esto puede ser utilizado para episodic y matins normal. Podrían ser labels adicionales, uno para cuando es modificada solo temporalmente, y otro cuando es modificada en palabras y temporalmente.

## Current Best Setup for MAX SR Unseen


- Best results were VAL UNSEEN: SR 47.5X, SPL: 44.8 SPL
- In the original paper they got SR: 47.98 SPL 44.1 SPL
- And their  baseline was SR: 46.40, SPL: 42.89

The setup to obtain those results was something like this:

80K iters 0.0001, matins 1
40k iters 0.00005, matins 1
UNK iters 0.0001, matins 5 >>> matins 2

Im currently trying to replicate those results with 
120k iters 0.0001 LR on the aux task, and 1 weight ponderation in the total loss. And then I have to see if its a good idea to increment the weight ponderation of the aux task to a greater one, or if just more iterations would do the trick

## Considerations




