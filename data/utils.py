from torch.utils.data import Dataset
import torch


class SequenceGenerator(Dataset):
    def __init__(self, sequence_size, dataset_size) -> None:
        super().__init__()
        self.sequence_size = sequence_size
        self.dataset_size = dataset_size
        assert self.dataset_size > 4, "Inadequate data points!"
        assert self.sequence_size > 2, "Sequence length should be larger"
        self.get_index = lambda: torch.randint(
            low=0, high=self.sequence_size, size=(1,)
        )

    def generate_sample(self):
        sample_input = torch.rand(2, self.sequence_size)
        target = 0
        sample_input[1] = (sample_input[1]) * 0

        index1 = self.get_index()
        target += sample_input[0][index1]
        sample_input[1][index1] = 1

        while True:
            index2 = self.get_index()
            if index1 != index2:
                break

        target += sample_input[0][index2]
        sample_input[1][index2] = 1

        return sample_input.T, target

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, _):
        return self.generate_sample()
