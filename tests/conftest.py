import pytest

from synthetic import SyntheticObject, make_blob_object


@pytest.fixture(scope="session")
def blob(tmp_path_factory: pytest.TempPathFactory) -> SyntheticObject:
    return make_blob_object(tmp_path_factory.mktemp("mesh"))
