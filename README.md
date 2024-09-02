En 'lda_cv.py' se realizo cross validation para determinar los mejores parametros, obteniendo:
    random_state=42, //semilla con la que se trabajo
    num_topics=15,
    passes=30,
    alpha='asymmetric',
    eta=0.9

    para llegar a esto se trabajo como base con:
        parameters = {'num_topics': [5, 10, 15, 20, 25], 'alpha': [falta, 'asymmetric', falta], 'eta': [0.85, 0.9, 0.95], 'passes':[15, 20, 25, 30, 35, 40, 45, 50]}

En 'lda_v2' se calculo la perplexity y varios tipos de coherence ('u_mass', 'c_v', 'c_w2v', 'c_uci', 'c_npmi') utilizando los parametros con mejor resultados obtenidos de 'lda_cv'.

# ejecucion
    1_ Intalar todo lo necesario
    2_ Correr con 'python lda_v2.py'