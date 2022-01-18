## Funcionamiento de Algoritmo Codigo
- Importante mencionar que matching y las tareas asociadas solo se entrenan durante
teacher forcing. El rollout se ejecuta una vez en teacher forcing y otra en "sample" que sería argmax.

### Lineas de código importantes
    # EMPEZANDO EL ROLLOUT (1 iteracion del batch)
    seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
    # _sort_batch agrupa en seq (un tensor), las instrucciones (codificadas)
    ctx, h_t, c_t = self.encoder(seq, seq_lengths)
    # se las pasa al encoder y genera el vector ctx de atencion sobre las instrucciones

    # luego se entra al episodio
    for t in range(self.episode_len):
    # se le pasa al decoder
        h_t, c_t, logit, h1 = self.decoder(input_a_t,candidate_feat,
                                               h1, c_t,
                                               ctx, ctx_mask,feature=f_t,
                                               sparseObj=sparseObj,denseObj=denseObj,
                                               ObjFeature_mask=ObjFeature_mask,
                                               already_dropfeat=(speaker is not None))
    # al finalizar el episodio se ejecuta matching original
    # que utiliza ctx y v_ctx (atención de la instruccion y atencion visual)
    # existe otro vector vl_ctx de la atención crossmodal, el paper menciona
    # que utilizan vl_ctx, pero en el código del repo utilizan v_ctx.
    
### Lineas de código importantes para la modificacion de matching
    # EMPEZANDO EL ROLLOUT
    # crear instrucción falsa (se hace para cada iter)
    # siguiendo los mismos pasos de la instrucción original
    seq_fake, _, seq_lengths_fake, _ = self._sort_batch_fake_instruction(obs)
    with torch.no_grad():
        ctx_fake, _, _ = self.encoder(seq_fake, seq_lengths_fake)

    # El algoritmo para matins
    # utiliza v_ctx al igual que matching original, hay que probar vl_ctx
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



## Cosas por aclarar
- Rendimiento de vl_ctx v/s v_ctx y cual hace más sentido en usar
- Que tipo de features de lenguaje se están usando, first features only.
Entender bien los slicing que hacen, ya que en el paper original se sugiere first + last features of language.
    l_ctx = ctx[:,0,:].detach() # FIRST FEATURES ONLY
    # l_ctx = torch.cat((ctx[:,0,:], ctx[:,-1,:]), dim=1).detach() # FIRST AND LAST
En mis experimentos utilicé solo first, el paper recomienda first + last.
Entender que significa first features v/s first + last.