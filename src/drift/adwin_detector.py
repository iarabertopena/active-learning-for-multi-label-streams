def drift_detection(dataset_name, window_size=100):

    print(f"\nDataset: {dataset_name}")

    # carregar dados
    X, Y = load_multilabel_dataset(dataset_name)

    # criando stream
    dummy_X = np.zeros((1, X.shape[1]))
    dummy_y = np.zeros(1)

    stream = NumpyStream(
        X=dummy_X,
        y=dummy_y
    )

    schema = stream.get_schema()

    # modelo
    model = BinaryRelevance(
        model_class=AdaptiveRandomForestClassifier,
        n_labels=Y.shape[1],
        schema=schema
    )

    # detector
    adwin = ADWIN()

    # métricas
    hamming_scores = []

    # janela móvel
    acc_window = []

    # timestamps de drift
    drift_points = []

    # loop stream
    for i in range(len(X)):

        x = X[i]
        y = Y[i]

        # previsão
        y_pred = model.predict(x)

        # erro
        ham = hamming_loss(y, y_pred)

        hamming_scores.append(ham)

        # detector
        adwin.add_element(ham)

        # verifica drift
        if adwin.detected_change():
            print(f"Drift detectado em {i}")
            drift_points.append(i)

        # média móvel
        start = max(0, i - window_size)

        acc_window.append(
            np.mean(hamming_scores[start:i+1])
        )

        # treino
        model.train(x, y)

    return acc_window, drift_points
