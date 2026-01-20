def test_smoke():
    assert True


def test_imports():
    try:
        import analyzer
    except ImportError:
        assert False, "Failed to import analyzer package"
