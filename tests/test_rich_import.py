def test_rich_importable():
    import importlib
    module = importlib.import_module("rich")
    assert module is not None
