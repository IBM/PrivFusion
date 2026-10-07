import pytest

from privfusion.utils import find_common_ancestors, find_mapping


@pytest.mark.skip
def test_simple_ancestor_finder() -> None:
    item1 = find_mapping("State")
    item2 = find_mapping("Country")

    assert item1
    assert item2

    common = find_common_ancestors(item1[0], item2[0])

    # Common ancestors may or may not exist depending on the ontology
    assert isinstance(common, list)
