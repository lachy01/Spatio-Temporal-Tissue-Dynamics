"""Division classifier comparison (instructions/division_classifier_comparison_spec.md).

Compares how well the count models (NB, Poisson mixture) and label-trained
logistic regressions predict which cells divide, on two separate scales: the
probability that a given label construction calls a cell cycling (label scale),
and the probability that the cell is in the cycling state (state scale).
"""
