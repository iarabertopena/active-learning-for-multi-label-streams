committee_models = [
    AdaptiveRandomForestClassifier,
    HoeffdingAdaptiveTree,
    StreamingRandomPatches
]

class Committee:

    def __init__(self, model_classes, n_labels, schema):

        self.models = []

        for model_class in model_classes:

            br_model = BinaryRelevance(
                model_class=model_class,
                n_labels=n_labels,
                schema=schema
            )

            self.models.append(br_model)

    def predict_all(self, x):

        predictions = []

        for model in self.models:

            predictions.append(
                model.predict(x)
            )

        return np.array(predictions)

    def predict_proba_all(self, x):

        probabilities = []

        for model in self.models:

            probabilities.append(
                model.predict_proba(x)
            )

        return probabilities

    def train(self, x, y, label_mask=None):

        for model in self.models:

            model.train(x, y, label_mask=label_mask)

def compute_vote_entropy(predictions):

    entropies = []

    n_members = predictions.shape[0]

    for label in range(predictions.shape[1]):

        votes = predictions[:, label]

        counts = np.bincount(
            votes,
            minlength=2
        )

        probs = counts / n_members

        probs = probs[probs > 0]

        entropy = -np.sum(
            probs * np.log2(probs)
        )

        entropies.append(entropy)

    return np.mean(entropies)

class QueryByCommittee(ActiveLearningStrategy):

    def __init__(self, threshold, warmup):

        super().__init__()

        self.threshold = threshold
        self.warmup = warmup

    def query(
        self,
        x=None,
        y_true=None,
        y_pred=None,
        probabilities=None,
        committee_predictions=None,
        committee_probabilities=None
    ):

        self.total_seen += 1

        # warm-up
        if self.total_seen <= self.warmup:

            self.total_queried += 1
            return True

        if committee_predictions is None:
            return False

        disagreement = compute_vote_entropy(committee_predictions)

        #print("disagreement_vote_entropy =", disagreement)

        if disagreement >= self.threshold:

            self.total_queried += 1

            return True

        return False
