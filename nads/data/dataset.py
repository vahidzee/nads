import torch


class DirectionalLinearDataset(torch.utils.data.Dataset):
    """
    Creates a linearly separable dataset along the provided direction
    This code is a torch only rewrite of the original:
        https://github.com/LTS4/neural-anisotropy-directions/blob/master/directional_bias.py
    """
    def __init__(
            self,
            direction,
            shape,
            num_samples=10000,
            sigma=3,
            epsilon=1,
    ):
        self.direction = direction
        self.num_samples = num_samples
        self.sigma = sigma
        self.epsilon = epsilon
        self.shape = shape
        self.data, self.targets = self._generate_dataset(self.num_samples)
        super().__init__()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target

    def __len__(self):
        return self.num_samples

    def _generate_dataset(self, n_samples):
        if n_samples > 1:
            data_plus = self._generate_samples(n_samples // 2 + n_samples % 2, 0).type(torch.float32)
            labels_plus = torch.zeros([n_samples // 2 + n_samples % 2]).type(torch.long)
            data_minus = self._generate_samples(n_samples // 2, 1).type(torch.float32)
            labels_minus = torch.ones([n_samples // 2]).type(torch.long)
            data = torch.cat([data_plus, data_minus])
            labels = torch.cat([labels_plus, labels_minus])
        else:
            data = self._generate_samples(1, 0).type(torch.float32)
            labels = torch.zeros(1).type(torch.long)

        return data, labels

    def _generate_samples(self, n_samples, label):
        data = self._generate_noise_floor(n_samples)
        sign = 1 if label == 0 else -1
        data = sign * self.epsilon / 2 * self.direction[None, :] + self._project_orthogonal(data)
        return data

    def _generate_noise_floor(self, n_samples):
        data = self.sigma * torch.randn(n_samples, *self.shape)

        return data

    def _project(self, x):
        proj_x = torch.reshape(x, [x.shape[0], -1]) @ torch.reshape(self.direction, [-1, 1])
        return proj_x[:, :, None, None] * self.direction[None, :]

    def _project_orthogonal(self, x):
        return x - self._project(x)

