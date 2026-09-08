"""A disabled ETF must not stop an otherwise valid daily price request."""


def test_zero_weight_fund_does_not_require_pre_inception_prices():
    from core_satellite_alpha import _core_tickers_for_config
    config = {'core_weights': {'SPY': 0, 'QQQ': 1, 'TQQQ': 0},
              'regime_preset': {regime: {'core_weights': {'QQQ': 1, 'TQQQ': 0}}
                                for regime in ('risk_on', 'neutral', 'risk_off')}}
    assert _core_tickers_for_config(config) == ['QQQ', 'SPY']
    # A fund held in even one regime must retain its price coverage requirement.
    config['regime_preset']['risk_on']['core_weights']['TQQQ'] = 0.1
    assert _core_tickers_for_config(config) == ['QQQ', 'SPY', 'TQQQ']
