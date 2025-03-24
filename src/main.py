"""
Module principal pour l'exécution de l'algorithme de trading sur produits dérivés
"""
import os
import sys
import logging
import argparse
import json
from datetime import datetime

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.data.market_data import MarketDataHandler
from src.execution.order_execution import OrderExecutionHandler
from src.strategies.gamma_scalping import GammaScalpingStrategy
from src.strategies.dispersion_trading import DispersionTradingStrategy
from src.strategies.risk_reversal import RiskReversalStrategy


def setup_logging(log_level='INFO', log_file=None):
    """
    Configure le système de logging.
    
    Args:
        log_level (str): Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Fichier de log
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configurer le niveau de log
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Niveau de log invalide: {log_level}')
    
    # Configurer le handler
    handlers = []
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handlers.append(logging.FileHandler(log_file))
    
    handlers.append(logging.StreamHandler())
    
    # Configurer le logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )
    
    logging.info(f"Logging initialisé au niveau {log_level}")


def parse_arguments():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Algorithme de trading sur produits dérivés')
    
    parser.add_argument('--symbol', type=str, default='SPX',
                        help='Symbole du sous-jacent à trader')
    parser.add_argument('--strategy', type=str, default='gamma_scalping',
                        choices=['gamma_scalping', 'dispersion', 'risk_reversal'],
                        help='Stratégie à utiliser')
    parser.add_argument('--backtest', action='store_true',
                        help='Exécuter en mode backtest plutôt qu\'en temps réel')
    parser.add_argument('--config', type=str, default='config.py',
                        help='Fichier de configuration alternatif')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Niveau de logging')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Fichier de log')
    
    return parser.parse_args()


def load_config(config_file):
    """
    Charge la configuration depuis un fichier.
    
    Args:
        config_file (str): Chemin vers le fichier de configuration
        
    Returns:
        dict: Configuration chargée
    """
    if not os.path.exists(config_file):
        logging.warning(f"Fichier de configuration {config_file} non trouvé, utilisation des valeurs par défaut")
        return CONFIG
    
    try:
        if config_file.endswith('.py'):
            # Import dynamique d'un module Python
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config = getattr(config_module, 'CONFIG')
        elif config_file.endswith('.json'):
            # Chargement d'un fichier JSON
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            logging.warning(f"Format de configuration non supporté: {config_file}, utilisation des valeurs par défaut")
            return CONFIG
        
        logging.info(f"Configuration chargée depuis {config_file}")
        return config
    except Exception as e:
        logging.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        return CONFIG


def init_data_provider(config):
    """
    Initialise le fournisseur de données.
    
    Args:
        config (dict): Configuration
        
    Returns:
        object: Fournisseur de données
    """
    provider_name = config.get('market_data', {}).get('provider', 'simulation')
    
    if provider_name == 'simulation':
        # En mode simulation, pas besoin de fournisseur externe
        return None
    
    # Ici, on pourrait implémenter la connexion à différents fournisseurs de données
    # comme Interactive Brokers, Alpha Vantage, etc.
    logging.warning(f"Fournisseur de données {provider_name} non implémenté, utilisation du mode simulation")
    return None


def create_strategy(strategy_name, config):
    """
    Crée une instance de stratégie en fonction du nom.
    
    Args:
        strategy_name (str): Nom de la stratégie
        config (dict): Configuration
        
    Returns:
        object: Instance de la stratégie
    """
    strategy_config = config.get('strategies', {}).get(strategy_name, {})
    
    if strategy_name == 'gamma_scalping':
        return GammaScalpingStrategy(strategy_config)
    elif strategy_name == 'dispersion':
        return DispersionTradingStrategy(strategy_config)
    elif strategy_name == 'risk_reversal':
        return RiskReversalStrategy(strategy_config)
    else:
        logging.error(f"Stratégie non reconnue: {strategy_name}")
        raise ValueError(f"Stratégie non reconnue: {strategy_name}")


def run_backtest(strategy, symbol, config):
    """
    Exécute un backtest de la stratégie.
    
    Args:
        strategy (object): Instance de la stratégie
        symbol (str): Symbole du sous-jacent
        config (dict): Configuration
        
    Returns:
        dict: Résultats du backtest
    """
    logging.info(f"Démarrage du backtest pour {symbol} avec la stratégie {strategy.__class__.__name__}")
    
    # Paramètres du backtest
    backtest_config = config.get('backtest', {})
    start_date = datetime.strptime(backtest_config.get('start_date', '2023-01-01'), '%Y-%m-%d')
    end_date = datetime.strptime(backtest_config.get('end_date', '2023-12-31'), '%Y-%m-%d')
    initial_capital = backtest_config.get('initial_capital', 1000000)
    
    # Initialiser les gestionnaires
    market_data = MarketDataHandler(config.get('market_data', {}))
    execution_handler = OrderExecutionHandler(config.get('execution', {}))
    
    # Récupérer les données historiques
    historical_data = market_data.get_historical_prices(symbol, start_date, end_date, '1d')
    
    if historical_data.empty:
        logging.error(f"Pas de données historiques pour {symbol} de {start_date} à {end_date}")
        return {'success': False, 'message': 'Pas de données historiques'}
    
    # Résultats du backtest
    results = {
        'symbol': symbol,
        'strategy': strategy.__class__.__name__,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'trades': [],
        'daily_pnl': [],
        'cumulative_pnl': [],
        'metrics': {}
    }
    
    # Simuler le backtest jour par jour
    current_capital = initial_capital
    for date, row in historical_data.iterrows():
        # Préparer les données de marché pour cette date
        market_data_snapshot = market_data.get_market_data_for_strategy(
            symbol, 
            strategy.__class__.__name__.lower().replace('strategy', '')
        )
        
        # Exécuter la stratégie pour cette date
        strategy_result = strategy.run(market_data_snapshot, execution_handler)
        
        # Enregistrer les résultats
        if strategy_result.get('success', False):
            trade = {
                'date': date,
                'action': strategy_result.get('action', 'none'),
                'pnl': strategy_result.get('current_pnl', 0)
            }
            results['trades'].append(trade)
            
            # Mettre à jour le capital
            current_capital += trade['pnl']
            
            # Enregistrer le PnL quotidien et cumulatif
            results['daily_pnl'].append({'date': date, 'pnl': trade['pnl']})
            results['cumulative_pnl'].append({'date': date, 'pnl': current_capital - initial_capital})
    
    # Calculer les métriques de performance
    if results['cumulative_pnl']:
        final_pnl = results['cumulative_pnl'][-1]['pnl']
        returns = final_pnl / initial_capital
        
        # Calculer d'autres métriques (Sharpe ratio, drawdown, etc.)
        # ...
        
        results['metrics'] = {
            'final_capital': initial_capital + final_pnl,
            'total_pnl': final_pnl,
            'return': returns,
            'num_trades': len(results['trades'])
            # Autres métriques...
        }
    
    logging.info(f"Backtest terminé avec un PnL de {results['metrics'].get('total_pnl', 0):.2f}")
    return results


def run_live_trading(strategy, symbol, config):
    """
    Exécute le trading en direct.
    
    Args:
        strategy (object): Instance de la stratégie
        symbol (str): Symbole du sous-jacent
        config (dict): Configuration
    """
    logging.info(f"Démarrage du trading en direct pour {symbol} avec la stratégie {strategy.__class__.__name__}")
    
    # Initialiser les gestionnaires
    data_provider = init_data_provider(config)
    market_data = MarketDataHandler(config.get('market_data', {}), data_provider)
    execution_handler = OrderExecutionHandler(config.get('execution', {}), data_provider)
    
    try:
        # Boucle de trading principale
        while True:
            # Récupérer les données de marché actuelles
            market_data_current = market_data.get_market_data_for_strategy(
                symbol, 
                strategy.__class__.__name__.lower().replace('strategy', '')
            )
            
            # Exécuter la stratégie
            strategy_result = strategy.run(market_data_current, execution_handler)
            
            if strategy_result.get('success', False):
                action = strategy_result.get('action', 'none')
                if action != 'none':
                    logging.info(f"Action exécutée: {action}")
                    # Enregistrer l'action dans une base de données ou un fichier
                    # ...
            
            # Attendre avant la prochaine itération
            # TODO: Implémenter une attente intelligente en fonction de la fréquence configurée
            import time
            time.sleep(60)  # 1 minute par défaut
            
    except KeyboardInterrupt:
        logging.info("Arrêt du trading en direct (interruption utilisateur)")
    except Exception as e:
        logging.error(f"Erreur lors du trading en direct: {str(e)}")
        raise


def main():
    """
    Fonction principale du programme.
    """
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer le logging
    setup_logging(args.log_level, args.log_file)
    
    # Charger la configuration
    config = load_config(args.config)
    
    # Créer la stratégie
    try:
        strategy = create_strategy(args.strategy, config)
    except ValueError as e:
        logging.error(str(e))
        return
    
    # Exécuter en mode backtest ou en direct
    if args.backtest:
        results = run_backtest(strategy, args.symbol, config)
        
        # Sauvegarder les résultats
        results_file = f"results_{args.symbol}_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(config.get('base', {}).get('results_dir', './results'), results_file), 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        logging.info(f"Résultats sauvegardés dans {results_file}")
    else:
        run_live_trading(strategy, args.symbol, config)


if __name__ == "__main__":
    main()
