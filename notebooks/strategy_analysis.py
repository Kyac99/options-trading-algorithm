"""
Ce script est conçu pour être exécuté dans un notebook Jupyter
afin de démontrer l'utilisation de l'algorithme de trading et
d'analyser les résultats des différentes stratégies.
"""

# Importer les bibliothèques nécessaires
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules du projet
from config import CONFIG
from src.strategies.gamma_scalping import GammaScalpingStrategy
from src.strategies.dispersion_trading import DispersionTradingStrategy
from src.strategies.risk_reversal import RiskReversalStrategy
from backtesting.backtester import BackTester

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("muted")

# Configurer les paramètres du backtest
backtest_config = {
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'initial_capital': 1000000,
    'commission_model': 'percentage',
    'commission_value': 0.0002,  # 0.02%
    'market_data': CONFIG.get('market_data', {}),
    'execution': CONFIG.get('execution', {}),
    'results_dir': './results'
}

# Fonction pour exécuter les backtests
def run_strategy_backtests(symbols, strategies_config):
    """
    Exécute des backtests pour plusieurs symboles et stratégies.
    
    Args:
        symbols (list): Liste de symboles à tester
        strategies_config (dict): Configuration des stratégies
        
    Returns:
        dict: Résultats des backtests
    """
    # Créer le répertoire des résultats s'il n'existe pas
    if not os.path.exists(backtest_config['results_dir']):
        os.makedirs(backtest_config['results_dir'])
    
    # Initialiser le backtester
    backtester = BackTester(backtest_config)
    
    # Résultats des backtests
    results = {}
    
    # Exécuter les backtests pour chaque symbole et stratégie
    for symbol in symbols:
        symbol_results = {}
        
        # Stratégie de Gamma Scalping
        if strategies_config.get('gamma_scalping', {}).get('active', False):
            print(f"Exécution du backtest de Gamma Scalping pour {symbol}...")
            strategy = GammaScalpingStrategy(strategies_config.get('gamma_scalping', {}))
            gamma_results = backtester.run_backtest(strategy, symbol)
            symbol_results['gamma_scalping'] = gamma_results
        
        # Stratégie de Dispersion Trading
        if strategies_config.get('dispersion', {}).get('active', False):
            print(f"Exécution du backtest de Dispersion Trading pour {symbol}...")
            strategy = DispersionTradingStrategy(strategies_config.get('dispersion', {}))
            dispersion_results = backtester.run_backtest(strategy, symbol)
            symbol_results['dispersion'] = dispersion_results
        
        # Stratégie de Risk Reversal
        if strategies_config.get('risk_reversal', {}).get('active', False):
            print(f"Exécution du backtest de Risk Reversal pour {symbol}...")
            strategy = RiskReversalStrategy(strategies_config.get('risk_reversal', {}))
            risk_reversal_results = backtester.run_backtest(strategy, symbol)
            symbol_results['risk_reversal'] = risk_reversal_results
        
        results[symbol] = symbol_results
    
    return results

# Fonction pour analyser les résultats
def analyze_results(results):
    """
    Analyse les résultats des backtests et affiche des graphiques et statistiques.
    
    Args:
        results (dict): Résultats des backtests
        
    Returns:
        DataFrame: Tableau de comparaison des performances
    """
    # Préparer les données pour la comparaison
    comparison_data = []
    
    for symbol, symbol_results in results.items():
        for strategy_name, strategy_results in symbol_results.items():
            metrics = strategy_results.get('metrics', {})
            
            strategy_data = {
                'Symbol': symbol,
                'Strategy': strategy_name,
                'Total Return (%)': metrics.get('total_return', 0) * 100,
                'Annualized Return (%)': metrics.get('annualized_return', 0) * 100,
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                'Win Rate (%)': metrics.get('win_rate', 0) * 100,
                'Profit Factor': metrics.get('profit_factor', 0),
                'Number of Trades': metrics.get('num_trades', 0)
            }
            
            comparison_data.append(strategy_data)
    
    # Créer un DataFrame pour la comparaison
    comparison_df = pd.DataFrame(comparison_data)
    
    # Afficher le tableau de comparaison
    display(comparison_df)
    
    # Créer un graphique de comparaison des courbes d'équité
    plt.figure(figsize=(14, 8))
    
    for symbol, symbol_results in results.items():
        for strategy_name, strategy_results in symbol_results.items():
            if not strategy_results.get('equity_curve', []):
                continue
                
            equity_df = pd.DataFrame(strategy_results['equity_curve'])
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df.set_index('date', inplace=True)
            
            # Normaliser à 100 pour une comparaison plus facile
            initial_equity = strategy_results['initial_capital']
            equity_df['normalized'] = equity_df['equity'] * 100 / initial_equity
            
            plt.plot(equity_df.index, equity_df['normalized'], linewidth=2, label=f"{symbol} - {strategy_name}")
    
    plt.title('Comparaison des stratégies - Courbes d\'équité normalisées')
    plt.xlabel('Date')
    plt.ylabel('Équité (base 100)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Créer un graphique de comparaison des métriques clés
    metrics_to_plot = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        # Créer un barplot pour la métrique
        ax = sns.barplot(x='Symbol', y=metric, hue='Strategy', data=comparison_df)
        
        # Ajouter les valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        
        plt.title(f'Comparaison des stratégies - {metric}')
        plt.xlabel('Symbole')
        plt.ylabel(metric)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    return comparison_df

# Fonction pour analyser une stratégie spécifique en détail
def analyze_strategy_details(results, symbol, strategy_name):
    """
    Analyse en détail les résultats d'une stratégie spécifique.
    
    Args:
        results (dict): Résultats des backtests
        symbol (str): Symbole à analyser
        strategy_name (str): Nom de la stratégie à analyser
    """
    if symbol not in results or strategy_name not in results[symbol]:
        print(f"Aucun résultat trouvé pour {symbol} - {strategy_name}")
        return
    
    strategy_results = results[symbol][strategy_name]
    
    # Afficher les métriques de performance
    metrics = strategy_results.get('metrics', {})
    print(f"=== Métriques de performance pour {symbol} - {strategy_name} ===")
    print(f"Rendement total: {metrics.get('total_return', 0) * 100:.2f}%")
    print(f"Rendement annualisé: {metrics.get('annualized_return', 0) * 100:.2f}%")
    print(f"Ratio de Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Drawdown maximum: {metrics.get('max_drawdown', 0) * 100:.2f}%")
    print(f"Taux de gain: {metrics.get('win_rate', 0) * 100:.2f}%")
    print(f"Facteur de profit: {metrics.get('profit_factor', 0):.2f}")
    print(f"Nombre de transactions: {metrics.get('num_trades', 0)}")
    print("")
    
    # Préparer les données
    if strategy_results.get('equity_curve', []):
        equity_df = pd.DataFrame(strategy_results['equity_curve'])
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        daily_pnl_df = pd.DataFrame(strategy_results['daily_pnl'])
        daily_pnl_df['date'] = pd.to_datetime(daily_pnl_df['date'])
        daily_pnl_df.set_index('date', inplace=True)
        
        # 1. Courbe d'équité
        plt.figure(figsize=(14, 7))
        plt.plot(equity_df.index, equity_df['equity'], linewidth=2)
        
        # Ajouter une ligne pour l'investissement initial
        plt.axhline(y=strategy_results['initial_capital'], color='r', linestyle='--', alpha=0.7)
        
        plt.title(f'Courbe d\'équité - {symbol} - {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Capital (€)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # 2. PnL quotidien
        plt.figure(figsize=(14, 7))
        plt.bar(daily_pnl_df.index, daily_pnl_df['pnl'], width=0.8, alpha=0.7,
                color=[('green' if x > 0 else 'red') for x in daily_pnl_df['pnl']])
        plt.title(f'PnL Quotidien - {symbol} - {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('PnL (€)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # 3. Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        
        plt.figure(figsize=(14, 7))
        plt.fill_between(equity_df.index, 0, equity_df['drawdown'], color='red', alpha=0.3)
        plt.plot(equity_df.index, equity_df['drawdown'], color='red', linewidth=1)
        plt.title(f'Drawdown - {symbol} - {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # 4. Distribution des rendements quotidiens
        equity_df['daily_return'] = equity_df['equity'].pct_change().fillna(0)
        
        plt.figure(figsize=(14, 7))
        sns.histplot(equity_df['daily_return'] * 100, bins=50, kde=True)
        plt.title(f'Distribution des rendements quotidiens - {symbol} - {strategy_name}')
        plt.xlabel('Rendement quotidien (%)')
        plt.ylabel('Fréquence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 5. Analyse des trades
        if strategy_results.get('trades', []):
            trades_df = pd.DataFrame(strategy_results['trades'])
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df['pnl'] = daily_pnl_df['pnl'].reindex(trades_df['date']).values
            
            # Afficher les 10 meilleures et pires transactions
            print("=== 10 meilleures transactions ===")
            display(trades_df.sort_values('pnl', ascending=False).head(10))
            
            print("=== 10 pires transactions ===")
            display(trades_df.sort_values('pnl', ascending=True).head(10))
            
            # Distribution des PnL des trades
            plt.figure(figsize=(14, 7))
            sns.histplot(trades_df['pnl'], bins=30, kde=True)
            plt.title(f'Distribution des PnL des transactions - {symbol} - {strategy_name}')
            plt.xlabel('PnL (€)')
            plt.ylabel('Fréquence')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Symboles à tester
    symbols = ['SPX', 'NDX', 'RUT']
    
    # Configuration des stratégies
    strategies_config = {
        'gamma_scalping': {
            'active': True,
            'rebalancing_frequency': '1h',
            'gamma_threshold': 0.05,
            'delta_hedge_threshold': 0.02,
            'position_size': 100
        },
        'dispersion': {
            'active': True,
            'correlation_threshold': 0.7,
            'volatility_lookback': 30,
            'position_size': 50
        },
        'risk_reversal': {
            'active': True,
            'skew_threshold': 0.15,
            'expiry_range': [30, 90],
            'position_size': 75
        }
    }
    
    # Exécuter les backtests
    backtest_results = run_strategy_backtests(symbols, strategies_config)
    
    # Analyser les résultats
    comparison = analyze_results(backtest_results)
    
    # Analyser en détail une stratégie spécifique
    analyze_strategy_details(backtest_results, 'SPX', 'gamma_scalping')
