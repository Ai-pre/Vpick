from src.validate_final_release import validate_release


def test_final_release_manifest_is_consistent() -> None:
    assert validate_release() == []
