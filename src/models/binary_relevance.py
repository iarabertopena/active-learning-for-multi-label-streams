class BinaryRelevance:

    def __init__(self, model_class, n_labels, schema):

        self.n_labels = n_labels
        self.models = []

        for _ in range(n_labels):

            model = model_class(
                schema=schema
            )

            self.models.append(model)


    # previsão
    def predict(self, x):

        predictions = []

        for model in self.models:

            temp_stream = NumpyStream(
                X=np.array([x]),
                y=np.array([0])
            )

            instance = temp_stream.next_instance()

            pred = model.predict(instance)

            if pred is None:
                pred = 0

            predictions.append(int(pred))

        return np.array(predictions)

    # treino
    def train(self, x, y):

        for j, model in enumerate(self.models):

            temp_stream = NumpyStream(
                X=np.array([x]),
                y=np.array([y[j]])
            )

            instance = temp_stream.next_instance()

            model.train(instance)
