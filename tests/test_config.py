from datetime import date


def test_config_loads_scale_from_env():
    from retail_gen.config import Config

    cfg = Config()
    assert cfg.scale == 0.001


def test_config_computes_customer_count():
    from retail_gen.config import Config

    cfg = Config()
    # 2M * 0.001 = 2000
    assert cfg.n_customers == 2000


def test_config_fixed_dimensions_ignore_scale():
    from retail_gen.config import Config

    cfg = Config()
    assert cfg.n_stores == 150
    assert cfg.n_products == 5000


def test_config_seed():
    from retail_gen.config import Config

    cfg = Config()
    assert cfg.seed == 42


def test_config_history_start():
    from retail_gen.config import Config

    cfg = Config()
    assert cfg.history_start == date(2023, 1, 1)
