from pydedisp import _libdedisp  # NOQA
import numpy as np


class Dedisp(object):
    def __init__(self, nchans, tsamp, fch1, foff, target="GPU") -> None:
        super().__init__()
        if target == "GPU":
            self._plan = _libdedisp.DedispPlan(nchans, tsamp, fch1, foff)
            self._plan.set_device(0)

    @property
    def max_delay(self):
        return self._plan.max_delay

    @property
    def channel_count(self):
        return self._plan.channel_count

    @property
    def dm_count(self):
        return self._plan.dm_count

    @property
    def tsamp(self):
        return self._plan.dt

    @property
    def foff(self):
        return self._plan.df

    @property
    def fch1(self):
        return self._plan.f0

    @property
    def dm_list(self):
        return self._plan.dm_list

    @dm_list.setter
    def dm_list(self, dm_list):
        if not isinstance(dm_list, np.ndarray):
            dm_list = np.array(dm_list, dtype=np.float32)
        if dm_list.ndim == 1 and dm_list.shape[0] > 0:
            self._plan.set_dm_list(dm_list)
        else:
            raise ValueError("dm_list should be 1d float array")

    @property
    def killmask(self):
        return self._plan.killmask

    @killmask.setter
    def killmask(self, killmask):
        if not isinstance(killmask, np.ndarray):
            killmask = np.array(killmask, dtype=np.int32)
        if killmask.ndim == 1 and killmask.shape[0] == self.channel_count:
            self._plan.set_killmask(killmask)
        else:
            raise ValueError("killmask should be 1d float array of length nchans")

    def generate_dm_list(self, dm_min, dm_max, width_int, dm_tol):
        self._plan.generate_dm_list(dm_min, dm_max, width_int, dm_tol)

    def dmt_transform(self, data, in_nbits=8):
        if self.dm_count == 0:
            raise ValueError("Please set or generate dm_list first")

        nsamps = data.shape[1]
        assert data.shape[0] == self.channel_count, "input data should have nchans"
        assert in_nbits in {1, 2, 4, 8, 16, 32}

        nsamps_out = nsamps - self.max_delay
        dmt = np.zeros(nsamps_out * self.dm_count, dtype=float)

        self._plan.execute(nsamps, data, in_nbits, dmt, 0)
        return dmt.reshape(self.dm_count, nsamps_out)
