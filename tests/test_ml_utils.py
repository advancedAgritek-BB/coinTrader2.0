from crypto_bot.utils.ml_utils import ML_AVAILABLE


def test_ml_available_is_bool():
    assert isinstance(ML_AVAILABLE, bool)
