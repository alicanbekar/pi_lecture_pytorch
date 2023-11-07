# ### Dataset Explanation
# 
# 1. **__init__ Method**: This method initializes the dataset with necessary parameters. It also reads data from the HDF5 file and calculates the normalization statistics.
# 2. **__len__ Method**: Returns the length of the dataset, which varies based on whether it's in training, validation, or testing mode.
# 3. **__getitem__ Method**: Provides a way to access individual data items. This method reads data for a given index, normalizes it, and returns it in the required format.
# 4. **get_initial_conditions Method**: Returns the initial conditions of the simulation, useful for usupervised learning tasks.
# 
# With this custom dataset, we've built the foundation for reading and preprocessing the SW simulation data. In the next sections, we'll build our model, loss function, and the training loop to complete our PyTorch project.
# 


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


def import_data(datafile,
                dtype=torch.float32,
                train_frac=0.5,
                valid_frac=0.2,
                normalize=True,
                max_sims=None):
    with np.load(datafile) as fid:
        zeta, v = fid['zeta'], fid['v']

    if max_sims is not None:
        zeta, v = zeta[:max_sims], v[:max_sims]

    # divide experiments (not time points) randomly into train/test/val
    n_exp = zeta.shape[0]
    exp_order = np.random.permutation(n_exp)

    n_train = int(n_exp * train_frac)
    n_val = int(n_exp * valid_frac)

    stats = dict(mz=zeta[exp_order[:n_train]].mean(),
                 sdz=zeta[exp_order[:n_train]].std(),
                 mv=v[exp_order[:n_train]].mean(),
                 sdv=v[exp_order[:n_train]].std())
    if normalize:
        zeta, v = (zeta - stats['mz']) / stats['sdz'], (v - stats['mv']) / stats['sdv']

    train_data = make_dataset(zeta[exp_order[:n_train]], v[exp_order[:n_train]], dtype=dtype)

    val_data = make_dataset(zeta[exp_order[n_train:n_train + n_val]], v[exp_order[n_train:n_train + n_val]],
                            dtype=dtype)
    test_data = make_dataset(zeta[exp_order[n_train + n_val:]], v[exp_order[n_train + n_val:]], dtype=dtype)

    return train_data, val_data, test_data, stats


def make_dataset(zeta, v, dtype=torch.float32):
    zeta_in = zeta[:, :-1, :].reshape(-1, zeta.shape[-1])
    zeta_out = zeta[:, 1:, :].reshape(-1, zeta.shape[-1])

    v_in = v[:, :-1, :].reshape(-1, v.shape[-1])
    v_out = v[:, 1:, :].reshape(-1, v.shape[-1])

    tensors = [torch.tensor(x, dtype=dtype).unsqueeze(1) for x in [zeta_in, zeta_out, v_in, v_out]]
    return TensorDataset(*tensors)


class SWDataset(Dataset):
    """A custom dataset to read the shallow water (SW) simulation data."""

    def __init__(
            self,
            file_path,
            order=1,
            numtime=1200,
            mode="train",
            train_frac=0.5,
            valid_frac=0.2,
            normalize=True
    ):
        """
        Initialize the dataset.

        Parameters:
            - file_path (str): Path to the data file.
            - order (int): Autoregressive model order.
            - numtime (int): Total number of time steps in the data.
            - mode (str): Mode of operation ("train", "valid", "test").
            - train_frac (float): Fraction of data to use for training.
            - valid_frac (float): Fraction of data to use for validation.
        """
        assert mode in [
            "train",
            "valid",
            "test",
        ], "Mode should be either 'train', 'valid', or 'test'"

        super(SWDataset, self).__init__()

        self.file_path = file_path
        self.order = order
        self.numtime = numtime
        self.mode = mode
        self.normalize = normalize

        # Determine split indices based on dataset size and provided fractions
        total_samples = self.numtime - 1 - self.order
        self.train_end = int(train_frac * total_samples)
        self.valid_end = self.train_end + int(valid_frac * total_samples)

        # Lists to store elevation and velocity data for normalization
        zetas = []
        velocities = []

        # Read data from HDF5 file
        with h5py.File(self.file_path, "r") as hdf_file:
            self.init_zeta = torch.tensor(hdf_file[f"timestep_0"]['elevation'][:])
            self.init_vel = torch.tensor(hdf_file[f"timestep_0"]['velocity'][:])
            for index in range(self.numtime):
                zetas.append(hdf_file[f"timestep_{index}"]["elevation"][:])
                velocities.append(hdf_file[f"timestep_{index}"]["velocity"][:])

        # Convert list to numpy array for easier operations
        zetas = np.array(zetas)
        velocities = np.array(velocities)

        # Calculate statistics (mean, std, min, max) for (optional) normalization
        self.zeta_min = zetas.min()
        self.zeta_max = zetas.max()
        self.velocity_min = velocities.min()
        self.velocity_max = velocities.max()

    def __len__(self):
        """Return the length of the dataset based on mode."""
        if self.mode == "train":
            return self.train_end
        elif self.mode == "valid":
            return self.valid_end - self.train_end
        else:  # mode is 'test'
            return self.numtime - 1 - self.order - self.valid_end

    def normalize_vals(self, z, v):
        if self.normalize:
            z = (z - self.zeta_min) / (self.zeta_max - self.zeta_min)
            v = 2.0 * (v - self.velocity_min) / (self.velocity_max - self.velocity_min) - 1.0
        return z, v

    def denormalize_vals(self, zeta, velocity):
        if self.normalize:
            zeta = zeta * (self.zeta_max - self.zeta_min) + self.zeta_min
            velocity = (velocity + 1.0) * 0.5 * (self.velocity_max - self.velocity_min) + self.velocity_min
        return zeta, velocity

    def __getitem__(self, index):
        """Return the data at the given index."""
        # Adjust the index based on mode (train/valid/test)
        if self.mode == "valid":
            index += self.train_end
        elif self.mode == "test":
            index += self.valid_end

        zetas, velocities = [], []

        # Read data for the given index from the HDF5 file
        with h5py.File(self.file_path, "r") as hdf_file:
            for i in range(self.order):
                # Normalize Elevation and velocity using min-max scaling
                zeta = hdf_file[f"timestep_{index + i}"]["elevation"][:]
                vel = hdf_file[f"timestep_{index + i}"]["velocity"][:]
                zeta, vel = self.normalize_vals(zeta, vel)

                # Add an extra dimension to match the expected input shape
                zetas.append(zeta[None, :])
                velocities.append(vel[None, :])

            # Extract and normalize target data for the given index
            target_zeta = hdf_file[f"timestep_{index + self.order}"]["elevation"][:]
            target_vel = hdf_file[f"timestep_{index + self.order}"]["velocity"][:]
            target_zeta, target_vel = self.normalize_vals(target_zeta, target_vel)

        # Concatenate the input data
        input_zeta = np.concatenate(zetas, axis=0)
        input_vel = np.concatenate(velocities, axis=0)

        # Convert input and target data to PyTorch tensors
        return (
            torch.tensor(input_zeta, dtype=torch.float32),
            torch.tensor(input_vel, dtype=torch.float32),
        ), (
            torch.tensor(
                target_zeta[None, :], dtype=torch.float32
            ),  # Add an extra dimension
            torch.tensor(
                target_vel[None, :], dtype=torch.float32
            ),  # Add an extra dimension
        )

    def get_initial_conditions(self):
        return self.init_zeta.unsqueeze(0).unsqueeze(0), self.init_vel.unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    datafile = 'data.npz'
    train_data, val_data, test_data, stats = import_data(datafile)
    with np.load(datafile) as fid:
        nsteps = fid['zeta'].shape[1]
