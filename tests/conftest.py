import pytest


@pytest.fixture(scope="class", autouse=True)
def header():
    info = {}
    info["nchans"] = 1024
    info["tsamp"] = 0.000064
    info["fch1"] = 920
    info["foff"] = -200 / info["nchans"]
    return info
