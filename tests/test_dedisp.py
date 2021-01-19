import numpy as np
from pydedisp import Dedisp


class TestDedisp:
    def test_initialization(self, header):
        mydedisp = Dedisp(
            header["nchans"], header["tsamp"], header["fch1"], header["foff"]
        )
        assert mydedisp.max_delay == 0
        assert mydedisp.channel_count == header["nchans"]
        assert mydedisp.dm_count == 0
        np.testing.assert_allclose(mydedisp.tsamp, header["tsamp"])
        assert mydedisp.fch1 == header["fch1"]
        assert mydedisp.foff == header["foff"]
        assert mydedisp.dm_list.size == 0
        assert mydedisp.killmask.size == header["nchans"]
        np.testing.assert_array_equal(
            np.ones(header["nchans"], dtype=np.int32), mydedisp.killmask
        )

    def test_setters(self, header):
        mydedisp = Dedisp(
            header["nchans"], header["tsamp"], header["fch1"], header["foff"]
        )
        dm_list = np.arange(100, dtype=np.float32)
        mydedisp.dm_list = dm_list
        assert mydedisp.dm_count == 100
        assert mydedisp.max_delay == 4792
        np.testing.assert_array_equal(dm_list, mydedisp.dm_list)
        killmask = np.zeros(header["nchans"], dtype=np.int32)
        mydedisp.killmask = killmask
        np.testing.assert_array_equal(killmask, mydedisp.killmask)

    def test_generate(self, header):
        mydedisp = Dedisp(
            header["nchans"], header["tsamp"], header["fch1"], header["foff"]
        )
        dm_list = np.array(
            [
                500.0,
                501.45917,
                502.92258,
                504.39023,
                505.86215,
                507.33838,
                508.81888,
                510.3037,
            ],
            dtype=np.float32,
        )
        mydedisp.generate_dm_list(500, 510, 40, 1.25)
        assert mydedisp.dm_count == 8
        assert mydedisp.max_delay == 24701
        np.testing.assert_array_equal(dm_list, mydedisp.dm_list)

    def test_transform(self, header):
        mydedisp = Dedisp(
            header["nchans"], header["tsamp"], header["fch1"], header["foff"]
        )
        dm_list = np.arange(100, dtype=np.float32)
        mydedisp.dm_list = dm_list

        np.random.seed(5)
        data = np.random.rand(header["nchans"], 16384)
        data = (data - 0.5) * 20 + 127.5
        data = data.astype(np.uint8)
        dmt = mydedisp.dmt_transform(data, in_nbits=8)
        assert dmt.shape == (mydedisp.dm_count, data.shape[1] - mydedisp.max_delay)


def test_benchmark_dedisp(header, benchmark):
    mydedisp = Dedisp(
        header["nchans"], header["tsamp"], header["fch1"], header["foff"]
    )
    dm_list = np.arange(3000, dtype=np.float32)
    mydedisp.dm_list = dm_list

    np.random.seed(5)
    data = np.random.rand(header["nchans"], 262144)
    data = (data - 0.5) * 20 + 127.5
    data = data.astype(np.uint8)
    dmt  = benchmark(mydedisp.dmt_transform, data, in_nbits=8)

    assert dmt.shape == (mydedisp.dm_count, data.shape[1] - mydedisp.max_delay)