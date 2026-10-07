import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import pytest  # noqa: E402

from kindify.data import synthetic_comments, validate  # noqa: E402
from kindify.experiment import train  # noqa: E402


@pytest.fixture(scope="session")
def comments():
    return validate(synthetic_comments(4000, seed=5))


@pytest.fixture(scope="session")
def trained(comments):
    return train(comments, seed=1)
