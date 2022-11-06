from data.utils import SequenceGenerator


def test_dataset():
    sg = SequenceGenerator(10, 100)
    sample, target = next(iter(sg))
    sample = sample.T
    non_zero_args = [sample[1] != 0]
    assert sum(sample[0][non_zero_args]) == target
