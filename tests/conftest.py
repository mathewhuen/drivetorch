import pytest


from utils import temp_cwd


@pytest.fixture(autouse=True)
def use_temp_cwd(tmp_path_factory):
    pth = tmp_path_factory.mktemp('temp')
    with temp_cwd(pth):
        yield
    return None

