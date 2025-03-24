# Configuration générale
CONFIG = {
    # Paramètres de base
    'base': {
        'log_level': 'INFO',
        'data_dir': './data',
        'results_dir': './results',
    },
    
    # Paramètres de connexion aux données de marché
    'market_data': {
        'provider': 'interactive_brokers',  # ou 'alphavantage', 'yahoo', etc.
        'api_key': 'YOUR_API_KEY',  # À remplacer par votre clé API
        'data_frequency': '1min',  # Fréquence des données: '1min', '5min', '1h', '1d', etc.
    },
    
    # Paramètres des stratégies
    'strategies': {
        'gamma_scalping': {
            'active': True,
            'rebalancing_frequency': '1h',  # Fréquence de rééquilibrage
            'gamma_threshold': 0.05,  # Seuil de gamma pour déclencher le scalping
            'delta_hedge_threshold': 0.02,  # Seuil de delta pour le hedging
            'position_size': 100,  # Taille de position standard
        },
        'dispersion_trading': {
            'active': False,
            'correlation_threshold': 0.7,  # Seuil de corrélation pour le trading de dispersion
            'volatility_lookback': 30,  # Période de lookback pour la volatilité (jours)
            'position_size': 50,
        },
        'risk_reversal': {
            'active': False,
            'skew_threshold': 0.15,  # Seuil de skew pour le Risk Reversal
            'expiry_range': [30, 90],  # Fourchette d'expiration en jours
            'position_size': 75,
        },
    },
    
    # Paramètres d'exécution
    'execution': {
        'method': 'vwap',  # 'vwap', 'twap', 'market_maker'
        'market_impact_model': 'linear',  # Modèle d'impact de marché
        'risk_aversion': 0.5,  # Coefficient d'aversion au risque [0,1]
        'slippage_model': 'percentage',  # 'percentage', 'fixed'
        'slippage_value': 0.0005,  # 0.05% de slippage
    },
    
    # Paramètres de backtesting
    'backtest': {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 1000000,  # Capital initial en EUR
        'commission_model': 'percentage',  # 'percentage', 'fixed', 'tiered'
        'commission_value': 0.0002,  # 0.02% de commission
    },
}