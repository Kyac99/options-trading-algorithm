"""
Module pour le backtesting des stratégies de trading
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.market_data import MarketDataHandler
from src.execution.order_execution import OrderExecutionHandler
from src.models.options_pricing import BlackScholesModel


class BackTester:
    """
    Classe de backtesting pour évaluer les stratégies de trading.
    """
    
    def __init__(self, config):
        """
        Initialise le backtester.
        
        Args:
            config (dict): Configuration pour le backtesting
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Paramètres du backtest
        self.start_date = datetime.strptime(config.get('start_date', '2023-01-01'), '%Y-%m-%d')
        self.end_date = datetime.strptime(config.get('end_date', '2023-12-31'), '%Y-%m-%d')
        self.initial_capital = config.get('initial_capital', 1000000)
        self.commission_model = config.get('commission_model', 'percentage')
        self.commission_value = config.get('commission_value', 0.0002)  # 0.02%
        
        # Initialiser les gestionnaires
        self.market_data = MarketDataHandler(config.get('market_data', {}))
        self.execution_handler = OrderExecutionHandler(config.get('execution', {}))
        
        # Données de backtest
        self.results = {}
        self.performance = {}
        
        # Créer le répertoire de résultats s'il n'existe pas
        self.results_dir = config.get('results_dir', './results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def run_backtest(self, strategy, symbol):
        """
        Exécute un backtest pour une stratégie donnée.
        
        Args:
            strategy (object): Instance de la stratégie à tester
            symbol (str): Symbole du sous-jacent
            
        Returns:
            dict: Résultats du backtest
        """
        self.logger.info(f"Démarrage du backtest pour {symbol} avec la stratégie {strategy.__class__.__name__}")
        strategy_name = strategy.__class__.__name__.lower().replace('strategy', '')
        
        # Récupérer les données historiques
        historical_prices = self.market_data.get_historical_prices(
            symbol, self.start_date, self.end_date, '1d'
        )
        
        if historical_prices.empty:
            self.logger.error(f"Pas de données historiques pour {symbol}")
            return {'success': False, 'message': 'Pas de données historiques'}
        
        # Résultats du backtest
        results = {
            'symbol': symbol,
            'strategy': strategy.__class__.__name__,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'trades': [],
            'positions': [],
            'daily_pnl': [],
            'equity_curve': [],
            'metrics': {}
        }
        
        # Initialiser le capital et les positions
        current_capital = self.initial_capital
        current_positions = []
        
        # Simuler le backtest jour par jour
        for date_index, row in historical_prices.iterrows():
            date = pd.to_datetime(date_index)
            
            # Préparer les données de marché pour cette date
            market_data_snapshot = self._prepare_market_data(symbol, date, strategy_name)
            
            # Exécuter la stratégie pour cette date
            strategy_result = strategy.run(market_data_snapshot, self.execution_handler)
            
            # Mettre à jour les positions et calculer le PnL
            if strategy_result.get('success', False):
                # Enregistrer l'action
                action = strategy_result.get('action', 'none')
                if action != 'none':
                    trade = {
                        'date': date,
                        'action': action,
                        'details': strategy_result.get('details', {})
                    }
                    results['trades'].append(trade)
                
                # Mettre à jour les positions
                if 'positions' in strategy_result:
                    current_positions = strategy_result['positions']
                    results['positions'].append({
                        'date': date,
                        'positions': current_positions
                    })
                
                # Calculer le PnL
                pnl = strategy_result.get('current_pnl', 0)
                
                # Appliquer les commissions
                commissions = self._calculate_commissions(strategy_result)
                pnl -= commissions
                
                # Mettre à jour le capital
                current_capital += pnl
                
                # Enregistrer le PnL quotidien et la courbe d'équité
                results['daily_pnl'].append({
                    'date': date,
                    'pnl': pnl,
                    'commissions': commissions
                })
                
                results['equity_curve'].append({
                    'date': date,
                    'equity': current_capital
                })
            
            # Simuler le passage à la journée suivante
            # (dans un cas réel, on attendrait jusqu'au lendemain)
        
        # Calculer les métriques de performance
        self._calculate_performance_metrics(results)
        
        # Sauvegarder les résultats
        self._save_results(results, symbol, strategy_name)
        
        # Mettre à jour les résultats globaux
        self.results[strategy_name] = results
        
        self.logger.info(f"Backtest terminé pour {symbol} avec la stratégie {strategy.__class__.__name__}")
        
        return results
    
    def _prepare_market_data(self, symbol, date, strategy_name):
        """
        Prépare les données de marché pour une date spécifique.
        
        Args:
            symbol (str): Symbole du sous-jacent
            date (datetime): Date pour laquelle préparer les données
            strategy_name (str): Nom de la stratégie
            
        Returns:
            dict: Données de marché
        """
        # Dans un cas réel, on utiliserait des données historiques complètes
        # Ici, on simule les données de marché pour la stratégie
        
        # Récupérer le prix du sous-jacent à cette date
        historical_prices = self.market_data.get_historical_prices(
            symbol, date, date + timedelta(days=1), '1d'
        )
        
        if historical_prices.empty:
            self.logger.warning(f"Pas de données pour {symbol} à la date {date}")
            # Utiliser un prix simulé
            price = 100.0  # Valeur par défaut
        else:
            price = historical_prices.iloc[0]['close']
        
        # Créer un timestamp à l'ouverture du marché pour cette date
        timestamp = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=9, minutes=30)
        
        # Générer une chaîne d'options synthétique pour cette date
        options_chain = self.market_data._generate_synthetic_option_chain(symbol)
        
        # Construire les données de marché selon la stratégie
        if strategy_name == 'gamma_scalping':
            market_data = {
                'timestamp': timestamp,
                'underlying_symbol': symbol,
                'underlying_price': price,
                'options_chain': options_chain,
                'risk_free_rate': 0.02
            }
        elif strategy_name == 'dispersion':
            # Simuler des composants pour la stratégie de dispersion
            components = self.market_data._simulate_index_components(symbol)
            
            market_data = {
                'timestamp': timestamp,
                'index': {
                    'symbol': symbol,
                    'price': price,
                    'options_chain': options_chain,
                    'implied_volatility': options_chain['implied_volatility'].mean(),
                    'options_prices': {row['symbol']: row['price'] for _, row in options_chain.iterrows()}
                },
                'components': components
            }
        elif strategy_name == 'risk_reversal':
            market_data = {
                'timestamp': timestamp,
                'underlying_symbol': symbol,
                'underlying_price': price,
                'options_chain': options_chain,
                'options_prices': {row['symbol']: row['price'] for _, row in options_chain.iterrows()},
                'risk_free_rate': 0.02
            }
        else:
            # Données génériques
            market_data = {
                'timestamp': timestamp,
                'underlying_symbol': symbol,
                'underlying_price': price,
                'options_chain': options_chain
            }
        
        return market_data
    
    def _calculate_commissions(self, strategy_result):
        """
        Calcule les commissions pour une exécution de stratégie.
        
        Args:
            strategy_result (dict): Résultat de l'exécution de la stratégie
            
        Returns:
            float: Montant des commissions
        """
        # Si pas de transactions, pas de commissions
        if not strategy_result.get('success', False) or strategy_result.get('action', 'none') == 'none':
            return 0.0
        
        # Extraire les détails des transactions
        transactions = []
        if 'results' in strategy_result:
            # Si plusieurs transactions dans un groupe
            for result in strategy_result['results']:
                if result.get('success', False):
                    transactions.append(result)
        else:
            # Si une seule transaction
            transactions.append(strategy_result)
        
        total_commissions = 0.0
        
        for transaction in transactions:
            if 'executed_price' in transaction and 'quantity' in transaction:
                price = transaction['executed_price']
                quantity = transaction['quantity']
                notional = price * quantity
                
                if self.commission_model == 'percentage':
                    commission = notional * self.commission_value
                elif self.commission_model == 'fixed':
                    commission = self.commission_value
                elif self.commission_model == 'tiered':
                    # Exemple de modèle de commission par paliers
                    if notional < 10000:
                        commission = notional * 0.0002  # 0.02%
                    elif notional < 100000:
                        commission = notional * 0.00015  # 0.015%
                    else:
                        commission = notional * 0.0001  # 0.01%
                else:
                    commission = 0.0
                
                total_commissions += commission
        
        return total_commissions
    
    def _calculate_performance_metrics(self, results):
        """
        Calcule les métriques de performance pour les résultats du backtest.
        
        Args:
            results (dict): Résultats du backtest
        """
        # Convertir les listes en DataFrame pour faciliter les calculs
        if not results['equity_curve']:
            results['metrics'] = {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': 0
            }
            return
        
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        daily_pnl_df = pd.DataFrame(results['daily_pnl'])
        daily_pnl_df['date'] = pd.to_datetime(daily_pnl_df['date'])
        daily_pnl_df.set_index('date', inplace=True)
        
        # Calculer les rendements quotidiens
        equity_df['daily_return'] = equity_df['equity'].pct_change().fillna(0)
        
        # Calculer le rendement total
        initial_equity = results['initial_capital']
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculer le rendement annualisé
        days = (results['end_date'] - results['start_date']).days
        annualized_return = (1 + total_return) ** (365 / max(1, days)) - 1
        
        # Calculer le ratio de Sharpe (en supposant un taux sans risque de 2%)
        risk_free_rate = 0.02
        daily_excess_return = equity_df['daily_return'] - risk_free_rate / 252
        sharpe_ratio = (252 ** 0.5) * daily_excess_return.mean() / daily_excess_return.std() if daily_excess_return.std() > 0 else 0
        
        # Calculer le drawdown maximum
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Statistiques des trades
        trade_stats = {}
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df['pnl'] = daily_pnl_df['pnl'].reindex(trades_df['date']).values
            
            # Taux de gain
            trades_df['is_win'] = trades_df['pnl'] > 0
            win_rate = trades_df['is_win'].mean()
            
            # Facteur de profit
            profit_sum = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            loss_sum = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
            
            trade_stats = {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'num_trades': len(trades_df)
            }
        else:
            trade_stats = {
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': 0
            }
        
        # Rassembler toutes les métriques
        results['metrics'] = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            **trade_stats
        }
    
    def _save_results(self, results, symbol, strategy_name):
        """
        Sauvegarde les résultats du backtest.
        
        Args:
            results (dict): Résultats du backtest
            symbol (str): Symbole du sous-jacent
            strategy_name (str): Nom de la stratégie
        """
        # Générer un nom de fichier avec un timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_{symbol}_{strategy_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convertir les dates en chaînes pour la sérialisation JSON
        results_json = {
            **results,
            'start_date': results['start_date'].strftime('%Y-%m-%d'),
            'end_date': results['end_date'].strftime('%Y-%m-%d'),
            'trades': [
                {**trade, 'date': trade['date'].strftime('%Y-%m-%d %H:%M:%S')}
                for trade in results['trades']
            ],
            'positions': [
                {**pos, 'date': pos['date'].strftime('%Y-%m-%d %H:%M:%S')}
                for pos in results['positions']
            ],
            'daily_pnl': [
                {**pnl, 'date': pnl['date'].strftime('%Y-%m-%d')}
                for pnl in results['daily_pnl']
            ],
            'equity_curve': [
                {**eq, 'date': eq['date'].strftime('%Y-%m-%d')}
                for eq in results['equity_curve']
            ]
        }
        
        # Sauvegarder en JSON
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        self.logger.info(f"Résultats sauvegardés dans {filepath}")
        
        # Générer et sauvegarder les graphiques
        self._generate_plots(results, symbol, strategy_name, timestamp)
    
    def _generate_plots(self, results, symbol, strategy_name, timestamp):
        """
        Génère et sauvegarde les graphiques des résultats du backtest.
        
        Args:
            results (dict): Résultats du backtest
            symbol (str): Symbole du sous-jacent
            strategy_name (str): Nom de la stratégie
            timestamp (str): Horodatage pour les noms de fichiers
        """
        if not results['equity_curve']:
            self.logger.warning("Pas de données pour générer des graphiques")
            return
        
        # Configurer le style des graphiques
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("muted")
        
        # Convertir les données en DataFrame
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        daily_pnl_df = pd.DataFrame(results['daily_pnl'])
        daily_pnl_df['date'] = pd.to_datetime(daily_pnl_df['date'])
        daily_pnl_df.set_index('date', inplace=True)
        
        # 1. Courbe d'équité
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['equity'], linewidth=2)
        plt.title(f'Courbe d\'équité - {symbol} - {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Capital (€)')
        plt.grid(True, alpha=0.3)
        
        # Formater l'axe y avec des séparateurs de milliers
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Formater l'axe x pour qu'il soit plus lisible
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Sauvegarder le graphique
        equity_plot_path = os.path.join(self.results_dir, f"equity_{symbol}_{strategy_name}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(equity_plot_path, dpi=150)
        plt.close()
        
        # 2. PnL quotidien
        plt.figure(figsize=(12, 6))
        plt.bar(daily_pnl_df.index, daily_pnl_df['pnl'], width=0.8, alpha=0.7,
                color=[('green' if x > 0 else 'red') for x in daily_pnl_df['pnl']])
        plt.title(f'PnL Quotidien - {symbol} - {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('PnL (€)')
        plt.grid(True, alpha=0.3)
        
        # Formater l'axe y avec des séparateurs de milliers
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Formater l'axe x pour qu'il soit plus lisible
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Sauvegarder le graphique
        pnl_plot_path = os.path.join(self.results_dir, f"pnl_{symbol}_{strategy_name}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(pnl_plot_path, dpi=150)
        plt.close()
        
        # 3. Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(equity_df.index, 0, equity_df['drawdown'], color='red', alpha=0.3)
        plt.plot(equity_df.index, equity_df['drawdown'], color='red', linewidth=1)
        plt.title(f'Drawdown - {symbol} - {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Formater l'axe x pour qu'il soit plus lisible
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Sauvegarder le graphique
        drawdown_plot_path = os.path.join(self.results_dir, f"drawdown_{symbol}_{strategy_name}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(drawdown_plot_path, dpi=150)
        plt.close()
        
        self.logger.info(f"Graphiques sauvegardés dans {self.results_dir}")
    
    def compare_strategies(self, strategies_results):
        """
        Compare les performances de différentes stratégies.
        
        Args:
            strategies_results (dict): Résultats des stratégies à comparer
            
        Returns:
            DataFrame: Tableau de comparaison
        """
        if not strategies_results:
            self.logger.warning("Pas de résultats de stratégies à comparer")
            return pd.DataFrame()
        
        # Préparer les données pour la comparaison
        comparison_data = []
        
        for strategy_name, results in strategies_results.items():
            metrics = results.get('metrics', {})
            
            strategy_data = {
                'Strategy': strategy_name,
                'Symbol': results.get('symbol', ''),
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
        
        # Sauvegarder la comparaison
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_path = os.path.join(self.results_dir, f"comparison_{timestamp}.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        self.logger.info(f"Comparaison des stratégies sauvegardée dans {comparison_path}")
        
        # Générer un graphique de comparaison
        self._generate_comparison_plot(strategies_results, timestamp)
        
        return comparison_df
    
    def _generate_comparison_plot(self, strategies_results, timestamp):
        """
        Génère un graphique comparant les courbes d'équité de différentes stratégies.
        
        Args:
            strategies_results (dict): Résultats des stratégies à comparer
            timestamp (str): Horodatage pour les noms de fichiers
        """
        plt.figure(figsize=(12, 6))
        
        for strategy_name, results in strategies_results.items():
            if not results.get('equity_curve', []):
                continue
                
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df.set_index('date', inplace=True)
            
            # Normaliser à 100 pour une comparaison plus facile
            initial_equity = results['initial_capital']
            equity_df['normalized'] = equity_df['equity'] * 100 / initial_equity
            
            plt.plot(equity_df.index, equity_df['normalized'], linewidth=2, label=strategy_name)
        
        plt.title('Comparaison des stratégies - Courbes d\'équité normalisées')
        plt.xlabel('Date')
        plt.ylabel('Équité (base 100)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Formater l'axe x pour qu'il soit plus lisible
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Sauvegarder le graphique
        comparison_plot_path = os.path.join(self.results_dir, f"comparison_plot_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(comparison_plot_path, dpi=150)
        plt.close()
        
        self.logger.info(f"Graphique de comparaison sauvegardé dans {comparison_plot_path}")
