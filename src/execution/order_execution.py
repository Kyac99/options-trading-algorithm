"""
Module pour l'exécution des ordres et la gestion des transactions
"""
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import time
import random


class OrderExecutionHandler:
    """
    Gestionnaire d'exécution des ordres pour les stratégies de trading.
    Supporte différents algorithmes d'exécution comme VWAP, TWAP, etc.
    """
    
    def __init__(self, config, broker_api=None):
        """
        Initialise le gestionnaire d'exécution des ordres.
        
        Args:
            config (dict): Configuration pour l'exécution des ordres
            broker_api: Connexion à l'API du broker (si None, mode simulation)
        """
        self.config = config
        self.broker_api = broker_api
        self.logger = logging.getLogger(__name__)
        
        # Paramètres d'exécution
        self.execution_method = config.get('method', 'vwap')
        self.risk_aversion = config.get('risk_aversion', 0.5)
        self.slippage_model = config.get('slippage_model', 'percentage')
        self.slippage_value = config.get('slippage_value', 0.0005)
        self.market_impact_model = config.get('market_impact_model', 'linear')
        
        # Historique des transactions
        self.transactions = []
        
        self.logger.info(f"Gestionnaire d'exécution initialisé avec méthode: {self.execution_method}")
    
    def place_order(self, symbol, order_type, quantity, price=None, limit_price=None, time_in_force='DAY'):
        """
        Place un ordre sur le marché, en mode réel ou simulation.
        
        Args:
            symbol (str): Symbole de l'instrument
            order_type (str): Type d'ordre ('BUY', 'SELL')
            quantity (int): Quantité à acheter/vendre
            price (float, optional): Prix de référence (requis pour les simulations)
            limit_price (float, optional): Prix limite pour les ordres limites
            time_in_force (str): Durée de validité de l'ordre
            
        Returns:
            dict: Résultat de l'exécution de l'ordre
        """
        if self.broker_api is None:
            # Mode simulation
            return self._simulate_order_execution(symbol, order_type, quantity, price)
        else:
            # Mode réel - intégration avec l'API du broker
            return self._execute_order_via_broker(symbol, order_type, quantity, limit_price, time_in_force)
    
    def _simulate_order_execution(self, symbol, order_type, quantity, reference_price):
        """
        Simule l'exécution d'un ordre sur le marché.
        
        Args:
            symbol (str): Symbole de l'instrument
            order_type (str): Type d'ordre ('BUY', 'SELL')
            quantity (int): Quantité à acheter/vendre
            reference_price (float): Prix de référence pour la simulation
            
        Returns:
            dict: Résultat simulé de l'exécution de l'ordre
        """
        if reference_price is None:
            self.logger.error("Prix de référence requis pour la simulation d'ordre")
            return {'success': False, 'message': 'Prix de référence manquant'}
        
        # Calculer le slippage
        slippage = self._calculate_slippage(reference_price, quantity, order_type)
        
        # Calculer l'impact de marché
        market_impact = self._calculate_market_impact(reference_price, quantity, order_type)
        
        # Prix d'exécution simulé
        if order_type == 'BUY':
            executed_price = reference_price * (1 + slippage + market_impact)
        else:  # 'SELL'
            executed_price = reference_price * (1 - slippage - market_impact)
        
        # Arrondir le prix à un niveau raisonnable (2 décimales)
        executed_price = round(executed_price, 2)
        
        # Simuler un délai d'exécution
        execution_delay = random.uniform(0.1, 1.0)  # Entre 0.1 et 1 seconde
        time.sleep(execution_delay)
        
        # Enregistrer la transaction
        transaction = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'order_type': order_type,
            'quantity': quantity,
            'reference_price': reference_price,
            'executed_price': executed_price,
            'slippage': slippage,
            'market_impact': market_impact,
            'execution_delay': execution_delay
        }
        self.transactions.append(transaction)
        
        self.logger.info(f"Ordre simulé: {order_type} {quantity} {symbol} @ {executed_price:.2f} (ref: {reference_price:.2f})")
        
        return {
            'success': True,
            'symbol': symbol,
            'order_type': order_type,
            'quantity': quantity,
            'executed_price': executed_price,
            'transaction_id': len(self.transactions)
        }
    
    def _execute_order_via_broker(self, symbol, order_type, quantity, limit_price=None, time_in_force='DAY'):
        """
        Exécute un ordre via l'API du broker.
        
        Args:
            symbol (str): Symbole de l'instrument
            order_type (str): Type d'ordre ('BUY', 'SELL')
            quantity (int): Quantité à acheter/vendre
            limit_price (float, optional): Prix limite pour les ordres limites
            time_in_force (str): Durée de validité de l'ordre
            
        Returns:
            dict: Résultat de l'exécution de l'ordre
        """
        try:
            # Construire les paramètres de l'ordre en fonction de l'API du broker
            order_params = {
                'symbol': symbol,
                'side': order_type,
                'quantity': quantity,
                'type': 'MARKET' if limit_price is None else 'LIMIT',
                'timeInForce': time_in_force
            }
            
            if limit_price is not None:
                order_params['price'] = limit_price
            
            # Exécuter l'ordre via l'API du broker
            order_result = self.broker_api.place_order(order_params)
            
            # Enregistrer la transaction
            transaction = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'order_type': order_type,
                'quantity': quantity,
                'executed_price': order_result.get('executed_price', None),
                'broker_order_id': order_result.get('order_id', None),
                'status': order_result.get('status', 'UNKNOWN')
            }
            self.transactions.append(transaction)
            
            self.logger.info(f"Ordre exécuté via broker: {order_type} {quantity} {symbol} (ID: {order_result.get('order_id')})")
            
            return {
                'success': True,
                'symbol': symbol,
                'order_type': order_type,
                'quantity': quantity,
                'executed_price': order_result.get('executed_price', None),
                'order_id': order_result.get('order_id', None),
                'transaction_id': len(self.transactions)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution de l'ordre via broker: {str(e)}")
            return {
                'success': False,
                'message': f"Erreur d'exécution: {str(e)}",
                'symbol': symbol,
                'order_type': order_type,
                'quantity': quantity
            }
    
    def _calculate_slippage(self, price, quantity, order_type):
        """
        Calcule le slippage en fonction du modèle configuré.
        
        Args:
            price (float): Prix de référence
            quantity (int): Quantité de l'ordre
            order_type (str): Type d'ordre ('BUY', 'SELL')
            
        Returns:
            float: Slippage calculé
        """
        if self.slippage_model == 'fixed':
            # Slippage fixe en points de base
            return self.slippage_value
        
        elif self.slippage_model == 'percentage':
            # Slippage en pourcentage du prix
            base_slippage = self.slippage_value
            
            # Ajustement en fonction de la quantité (plus la quantité est grande, plus le slippage est important)
            quantity_factor = 1.0 + (quantity / 1000) * 0.1  # Ajuster selon les besoins
            
            return base_slippage * quantity_factor
        
        elif self.slippage_model == 'dynamic':
            # Modèle dynamique qui varie avec la volatilité (à implémenter)
            # Ici, on pourrait utiliser des données de marché pour estimer la volatilité
            # et ajuster le slippage en conséquence
            base_slippage = self.slippage_value
            volatility_factor = 1.0  # À remplacer par un calcul basé sur la volatilité
            
            return base_slippage * volatility_factor
        
        else:
            self.logger.warning(f"Modèle de slippage inconnu: {self.slippage_model}, utilisation du modèle par défaut")
            return self.slippage_value
    
    def _calculate_market_impact(self, price, quantity, order_type):
        """
        Calcule l'impact de marché en fonction du modèle configuré.
        
        Args:
            price (float): Prix de référence
            quantity (int): Quantité de l'ordre
            order_type (str): Type d'ordre ('BUY', 'SELL')
            
        Returns:
            float: Impact de marché calculé
        """
        # Paramètre de sensibilité à l'impact (à calibrer selon le marché)
        impact_factor = 0.0001
        
        if self.market_impact_model == 'linear':
            # Modèle d'impact linéaire: impact proportionnel à la quantité
            return impact_factor * quantity / 100
        
        elif self.market_impact_model == 'square_root':
            # Modèle d'impact en racine carrée: souvent utilisé pour les marchés liquides
            return impact_factor * np.sqrt(quantity / 100)
        
        elif self.market_impact_model == 'custom':
            # Modèle personnalisé (à implémenter selon les besoins)
            # Par exemple, un modèle qui prend en compte la volatilité et la liquidité
            return impact_factor * quantity / 100  # Placeholder
        
        else:
            self.logger.warning(f"Modèle d'impact de marché inconnu: {self.market_impact_model}, utilisation du modèle par défaut")
            return impact_factor * quantity / 100
    
    def execute_vwap(self, symbol, order_type, total_quantity, price, duration_minutes=60, intervals=10):
        """
        Exécute un ordre en utilisant la stratégie VWAP (Volume-Weighted Average Price).
        Divise l'ordre en plusieurs tranches sur une période donnée.
        
        Args:
            symbol (str): Symbole de l'instrument
            order_type (str): Type d'ordre ('BUY', 'SELL')
            total_quantity (int): Quantité totale à acheter/vendre
            price (float): Prix de référence
            duration_minutes (int): Durée totale d'exécution en minutes
            intervals (int): Nombre de tranches
            
        Returns:
            dict: Résultat de l'exécution VWAP
        """
        self.logger.info(f"Début d'exécution VWAP: {order_type} {total_quantity} {symbol} sur {duration_minutes} minutes")
        
        # Calculer la taille des tranches et l'intervalle de temps
        interval_seconds = (duration_minutes * 60) / intervals
        results = []
        total_executed = 0
        avg_price = 0
        
        # Distribution des volumes pour chaque tranche (à adapter selon le profil de volume)
        # Par défaut: U-shape, plus de volume au début et à la fin
        volume_distribution = self._generate_vwap_distribution(intervals)
        
        for i in range(intervals):
            # Calculer la quantité pour cette tranche
            tranche_quantity = int(total_quantity * volume_distribution[i])
            if tranche_quantity < 1:
                tranche_quantity = 1
            
            # Ne pas dépasser la quantité totale
            if total_executed + tranche_quantity > total_quantity:
                tranche_quantity = total_quantity - total_executed
            
            if tranche_quantity <= 0:
                break
            
            # Exécuter la tranche
            result = self.place_order(symbol, order_type, tranche_quantity, price)
            
            if result['success']:
                total_executed += tranche_quantity
                avg_price += result['executed_price'] * tranche_quantity
                results.append(result)
            
            # Attendre jusqu'à la prochaine tranche
            if i < intervals - 1:
                time.sleep(interval_seconds)
        
        # Calculer le prix moyen pondéré
        if total_executed > 0:
            avg_price /= total_executed
        
        return {
            'success': total_executed > 0,
            'symbol': symbol,
            'order_type': order_type,
            'total_quantity': total_quantity,
            'executed_quantity': total_executed,
            'avg_price': avg_price,
            'tranches': len(results),
            'results': results
        }
    
    def execute_twap(self, symbol, order_type, total_quantity, price, duration_minutes=60, intervals=10):
        """
        Exécute un ordre en utilisant la stratégie TWAP (Time-Weighted Average Price).
        Divise l'ordre en tranches égales sur une période donnée.
        
        Args:
            symbol (str): Symbole de l'instrument
            order_type (str): Type d'ordre ('BUY', 'SELL')
            total_quantity (int): Quantité totale à acheter/vendre
            price (float): Prix de référence
            duration_minutes (int): Durée totale d'exécution en minutes
            intervals (int): Nombre de tranches
            
        Returns:
            dict: Résultat de l'exécution TWAP
        """
        self.logger.info(f"Début d'exécution TWAP: {order_type} {total_quantity} {symbol} sur {duration_minutes} minutes")
        
        # Calculer la taille des tranches et l'intervalle de temps
        tranche_quantity = max(1, total_quantity // intervals)
        interval_seconds = (duration_minutes * 60) / intervals
        results = []
        total_executed = 0
        avg_price = 0
        
        for i in range(intervals):
            # Pour la dernière tranche, prendre le reste
            if i == intervals - 1:
                tranche_quantity = total_quantity - total_executed
            
            if tranche_quantity <= 0:
                break
            
            # Exécuter la tranche
            result = self.place_order(symbol, order_type, tranche_quantity, price)
            
            if result['success']:
                total_executed += tranche_quantity
                avg_price += result['executed_price'] * tranche_quantity
                results.append(result)
            
            # Attendre jusqu'à la prochaine tranche
            if i < intervals - 1:
                time.sleep(interval_seconds)
        
        # Calculer le prix moyen pondéré
        if total_executed > 0:
            avg_price /= total_executed
        
        return {
            'success': total_executed > 0,
            'symbol': symbol,
            'order_type': order_type,
            'total_quantity': total_quantity,
            'executed_quantity': total_executed,
            'avg_price': avg_price,
            'tranches': len(results),
            'results': results
        }
    
    def _generate_vwap_distribution(self, intervals):
        """
        Génère une distribution de volume pour la stratégie VWAP.
        
        Args:
            intervals (int): Nombre de tranches
            
        Returns:
            list: Distribution de volume normalisée
        """
        # Distribution en forme de U pour simuler les volumes typiques de marché
        distribution = []
        
        for i in range(intervals):
            x = i / (intervals - 1)  # Normaliser entre 0 et 1
            # Fonction en U: ax^2 + b, où a > 0
            value = 2 * (x - 0.5) ** 2 + 0.5
            distribution.append(value)
        
        # Normaliser pour que la somme soit égale à 1
        total = sum(distribution)
        normalized = [x / total for x in distribution]
        
        return normalized
    
    def get_transaction_history(self, start_time=None, end_time=None, symbol=None):
        """
        Récupère l'historique des transactions avec filtrage optionnel.
        
        Args:
            start_time (datetime, optional): Heure de début du filtre
            end_time (datetime, optional): Heure de fin du filtre
            symbol (str, optional): Filtrer par symbole
            
        Returns:
            list: Transactions filtrées
        """
        filtered_transactions = self.transactions
        
        if start_time:
            filtered_transactions = [t for t in filtered_transactions if t['timestamp'] >= start_time]
        
        if end_time:
            filtered_transactions = [t for t in filtered_transactions if t['timestamp'] <= end_time]
        
        if symbol:
            filtered_transactions = [t for t in filtered_transactions if t['symbol'] == symbol]
        
        return filtered_transactions
    
    def get_execution_stats(self, start_time=None, end_time=None):
        """
        Calcule des statistiques sur l'exécution des ordres.
        
        Args:
            start_time (datetime, optional): Heure de début pour les statistiques
            end_time (datetime, optional): Heure de fin pour les statistiques
            
        Returns:
            dict: Statistiques d'exécution
        """
        transactions = self.get_transaction_history(start_time, end_time)
        
        if not transactions:
            return {
                'total_transactions': 0,
                'avg_slippage': 0,
                'avg_market_impact': 0,
                'avg_execution_delay': 0
            }
        
        # Calculer les moyennes
        total_slippage = sum(t.get('slippage', 0) for t in transactions)
        total_market_impact = sum(t.get('market_impact', 0) for t in transactions)
        total_execution_delay = sum(t.get('execution_delay', 0) for t in transactions)
        
        return {
            'total_transactions': len(transactions),
            'avg_slippage': total_slippage / len(transactions),
            'avg_market_impact': total_market_impact / len(transactions),
            'avg_execution_delay': total_execution_delay / len(transactions)
        }
