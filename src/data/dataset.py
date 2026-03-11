from pathlib import Path

import h5py
import numpy as np
import torch

from src.data.geometry import GEOMETRY
from src.data.utils import load_showers
from src.utils import get_logger

logger = get_logger()


def preprocess_geo(batchsize, geo, train_on):
    len_geo = len(train_on)
    condn = np.zeros((batchsize, len_geo + 1)).astype(np.float32)
    try:
        condn[:, train_on.index(geo)] = 1
    except ValueError:
        condn[:, -1] = 1  # adaptation
    return condn


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = list(range(10))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> int:
        return self.data[index]


class CaloShowerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path=None,
        files=None,
        extension=".h5",
        use_cond_info=True,
        max_num_showers=None,
        need_geo_condn=False,
        train_on=None,
        is_ccd=False,
        **kwargs
    ):
        if root_path is None and files is None:
            raise ValueError("Either root_path or files must be passed.")
        if root_path is not None and files is not None:
            raise ValueError("Cannot pass both root_path and files.")

        self.root_path = Path(root_path) if root_path is not None else None
        self.files = list(self.root_path.glob(f"*{extension}")) if files is None else files

        self.need_geo_condn = need_geo_condn
        self.train_on = train_on
        self.is_ccd = is_ccd
        self.use_cond_info = use_cond_info
        self.max_num_showers = max_num_showers

        total_num_data, shower_shape = self._get_total_data()

        self.showers = np.zeros([total_num_data] + list(shower_shape), dtype=np.float32)
        self.energy = np.zeros([total_num_data], dtype=np.float32)
        self.phi = np.zeros([total_num_data], dtype=np.float32)
        self.theta = np.zeros([total_num_data], dtype=np.float32)
        if self.need_geo_condn:
            self.geo = np.zeros([total_num_data, len(train_on) + 1], dtype=np.float32)

        start_idx = 0
        for file_item in self.files:
            if self.need_geo_condn:
                geo_str, file_path = file_item
                end_idx = load_showers(
                    file_path,
                    self.showers,
                    self.energy,
                    self.phi,
                    self.theta,
                    start_idx,
                    is_ccd=is_ccd,
                )
                self.geo[start_idx:end_idx] = preprocess_geo(end_idx - start_idx, geo_str, train_on)
            else:
                end_idx = load_showers(
                    file_item,
                    self.showers,
                    self.energy,
                    self.phi,
                    self.theta,
                    start_idx,
                    is_ccd=is_ccd,
                )
            start_idx = end_idx

        file_list = "\n".join(f"- {file}" for file in self.files)
        logger.info(f"Loaded {len(self.showers)} showers from the following files:\n{file_list}")

        if max_num_showers is not None:
            logger.info(f"Limiting the number of showers to {max_num_showers}.")
            self.showers = self.showers[:max_num_showers]
            self.energy = self.energy[:max_num_showers]
            self.phi = self.phi[:max_num_showers]
            self.theta = self.theta[:max_num_showers]
            if self.need_geo_condn:
                self.geo = self.geo[:max_num_showers]

        if is_ccd:
            self.showers = (
                self.showers
                .reshape(self.showers.shape[0], GEOMETRY.N_CELLS_Z, GEOMETRY.N_CELLS_PHI, GEOMETRY.N_CELLS_R)
                .transpose(0, 3, 2, 1)
                * 0.033
            )

    def _get_total_data(self):
        num_data = 0
        shower_shape = None
        for file_item in self.files:
            file_path = file_item[1] if self.need_geo_condn else file_item
            with h5py.File(file_path, "r") as f:
                num_data += len(f["showers"])
                if shower_shape is None:
                    shower_shape = f["showers"].shape[1:]
        return num_data, shower_shape

    def _torch(self, *xs):
        if len(xs) == 1:
            return torch.from_numpy(xs[0]).float()
        else:
            return tuple(self._torch(*x) if isinstance(x, tuple) else self._torch(x) for x in xs)

    def __len__(self):
        return len(self.showers)

    def __getitem__(self, idx):
        if self.use_cond_info:
            conditions = (
                self.energy[idx:idx + 1],
                self.phi[idx:idx + 1],
                self.theta[idx:idx + 1],
            )
            if self.need_geo_condn:
                conditions = conditions + (self.geo[idx:idx + 1],)
            return self._torch(self.showers[idx][None, ...], conditions)
        else:
            return self._torch(self.showers[idx][None, ...])
