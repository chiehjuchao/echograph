from overrides_ import overrides
import torch

from radgraph.allennlp.training.metrics.average import Average
from radgraph.allennlp.training.metrics.metric import Metric


@Metric.register("perplexity")
class Perplexity(Average):
    """
    Perplexity is a common metric used for evaluating how well a language model
    predicts a sample.

    Notes
    -----
    Assumes negative log likelihood loss of each batch (base e). Provides the
    average perplexity of the batches.
    """

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated perplexity.
        """
        average_loss = super().get_metric(reset)
        if average_loss == 0:
            perplexity = 0.0

        # Exponentiate the loss to compute perplexity
        perplexity = float(torch.exp(average_loss))

        return perplexity
