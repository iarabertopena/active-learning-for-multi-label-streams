def compute_uncertainty(probabilities):
    uncertainties = []

    for proba in probabilities:
        # Least Confidence
        confidence = np.max(proba)
        uncertainty = 1 - confidence
        uncertainties.append(uncertainty)

    result = np.max(uncertainties)

    return result

class UncertaintySampling(ActiveLearningStrategy):

    def __init__(self, threshold):

        super().__init__()

        self.threshold = threshold

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

        uncertainty = compute_uncertainty(
            probabilities
        )

        if uncertainty >= self.threshold:

            self.total_queried += 1

            return True

        return False
