"""
Module pour la stratégie de Dispersion Trading
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from src.models.options_pricing import BlackScholesModel


class DispersionTradingStrategy:
    """
    Stratégie de Dispersion Trading - Exploite les différences entre la volatilité implicite 
    d'un indice et celle de ses composants. L'approche consiste généralement à vendre des options 
    sur l'indice et à acheter des options sur ses composants (ou inversement) lorsqu'il y a une 
    anomalie de prix significative.
    """
    
    def __init__(self, config):
        """
        Initialise la stratégie de Dispersion Trading.
        
        Args:
            config (dict): Configuration de la stratégie
        """
        self.config = config
        self.bs_model = BlackScholesModel()
        self.logger = logging.getLogger(__name__)
        
        # Paramètres de la stratégie
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.volatility_lookback = config.get('volatility_lookback', 30)
        self.position_size = config.get('position_size', 50)
        
        # État de la stratégie
        self.positions = {
            'index_options': [],  # Options sur l'indice
            'component_options': [],  # Options sur les composants
        }
        self.dispersion_history = []
        self.active_trade = False
        
        self.logger.info("Stratégie de Dispersion Trading initialisée")
    
    def calculate_implied_correlation(self, index_iv, component_ivs, weights):
        """
        Calcule la corrélation implicite entre un indice et ses composants
        basée sur les volatilités implicites.
        
        Args:
            index_iv (float): Volatilité implicite de l'indice
            component_ivs (array): Volatilités implicites des composants
            weights (array): Poids des composants dans l'indice
            
        Returns:
            float: Corrélation implicite calculée
        """
        # Carré de la volatilité de l'indice
        index_variance = index_iv ** 2
        
        # Somme pondérée des variances des composants
        weighted_component_variance = sum([(w * iv) ** 2 for w, iv in zip(weights, component_ivs)])
        
        # Somme pondérée des covariances (si corrélation = 1)
        weighted_covariance_sum = 0
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                weighted_covariance_sum += 2 * weights[i] * weights[j] * component_ivs[i] * component_ivs[j]
        
        # Calcul de la corrélation implicite
        if weighted_covariance_sum > 0:
            implied_correlation = (index_variance - weighted_component_variance) / weighted_covariance_sum
            # Limiter la corrélation entre -1 et 1
            implied_correlation = max(-1.0, min(1.0, implied_correlation))
        else:
            implied_correlation = 0
            
        return implied_correlation
    
    def is_dispersion_opportunity(self, market_data):
        """
        Détermine s'il existe une opportunité de dispersion trading
        en analysant les volatilités implicites.
        
        Args:
            market_data (dict): Données de marché incluant l'indice et ses composants
            
        Returns:
            tuple: (bool, float) - Opportunité détectée et corrélation implicite
        """
        index_data = market_data['index']
        components_data = market_data['components']
        
        # Extraire les volatilités implicites
        index_iv = index_data['implied_volatility']
        component_ivs = [comp['implied_volatility'] for comp in components_data]
        component_weights = [comp['weight'] for comp in components_data]
        
        # Calculer la corrélation implicite
        implied_correlation = self.calculate_implied_correlation(
            index_iv, component_ivs, component_weights
        )
        
        # Enregistrer l'historique de dispersion
        self.dispersion_history.append({
            'timestamp': market_data['timestamp'],
            'implied_correlation': implied_correlation,
            'index_iv': index_iv,
            'avg_component_iv': np.average(component_ivs, weights=component_weights)
        })
        
        # Déterminer s'il y a une opportunité de trading
        correlation_anomaly = implied_correlation < self.correlation_threshold
        
        self.logger.info(f"Corrélation implicite: {implied_correlation:.4f}, Seuil: {self.correlation_threshold}")
        
        return correlation_anomaly, implied_correlation
    
    def find_optimal_options(self, options_chain, is_index=True):
        """
        Sélectionne les options optimales pour la stratégie de dispersion.
        
        Args:
            options_chain (DataFrame): Chaîne d'options disponibles
            is_index (bool): Indique si les options sont sur l'indice
            
        Returns:
            list: Liste des options optimales pour la stratégie
        """
        # Pour la stratégie de dispersion, nous préférons des options at-the-money (ATM)
        # Filtrer les options avec un delta proche de 0.5 (ou -0.5 pour les puts)
        atm_options = options_chain[
            (options_chain['option_type'] == 'call') & 
            (options_chain['delta'].between(0.45, 0.55)) |
            (options_chain['option_type'] == 'put') & 
            (options_chain['delta'].between(-0.55, -0.45))
        ]
        
        if atm_options.empty:
            self.logger.warning(f"Aucune option ATM trouvée ({'indice' if is_index else 'composants'})")
            return []
        
        # Sélectionner les options avec une expiration intermédiaire (1-3 mois)
        today = datetime.now()
        target_min_expiry = today + timedelta(days=30)
        target_max_expiry = today + timedelta(days=90)
        
        filtered_options = atm_options[
            (atm_options['expiry_date'] >= target_min_expiry) &
            (atm_options['expiry_date'] <= target_max_expiry)
        ]
        
        if filtered_options.empty:
            self.logger.warning("Aucune option avec l'expiration cible trouvée")
            # Utiliser les options ATM si aucune option ne correspond à l'expiration cible
            filtered_options = atm_options
        
        # Classer par liquidité (spread bid-ask minimum)
        filtered_options['spread_pct'] = (filtered_options['ask'] - filtered_options['bid']) / filtered_options['mid_price']
        sorted_options = filtered_options.sort_values('spread_pct')
        
        # Sélectionner les meilleures options
        best_options = sorted_options.head(3 if is_index else 10).to_dict('records')
        
        self.logger.info(f"Options optimales sélectionnées ({len(best_options)}) pour {'indice' if is_index else 'composants'}")
        return best_options
    
    def execute_dispersion_trade(self, market_data, execution_handler, implied_correlation):
        """
        Exécute un trade de dispersion en vendant des options sur l'indice
        et en achetant des options sur les composants (ou inversement).
        
        Args:
            market_data (dict): Données de marché actuelles
            execution_handler: Gestionnaire d'exécution des ordres
            implied_correlation (float): Corrélation implicite calculée
            
        Returns:
            dict: Résultat de l'exécution du trade
        """
        # Si un trade est déjà actif, ne pas en ouvrir un nouveau
        if self.active_trade:
            self.logger.info("Trade de dispersion déjà actif, pas de nouveau trade ouvert")
            return {'success': False, 'message': 'Trade déjà actif'}
        
        index_options_chain = market_data['index']['options_chain']
        components_data = market_data['components']
        
        # Trouver les options optimales
        index_options = self.find_optimal_options(index_options_chain, is_index=True)
        
        if not index_options:
            return {'success': False, 'message': 'Pas d\'options d\'indice appropriées trouvées'}
        
        # Dans une stratégie de dispersion classique, nous vendons des options sur l'indice
        # et achetons des options sur les composants lorsque la corrélation implicite est faible
        index_action = 'SELL'
        component_action = 'BUY'
        
        # Exécuter les ordres sur l'indice
        for opt in index_options:
            execution_result = execution_handler.place_order(
                symbol=opt['symbol'],
                order_type=index_action,
                quantity=self.position_size,
                price=opt['price']
            )
            
            if execution_result['success']:
                self.positions['index_options'].append({
                    'symbol': opt['symbol'],
                    'option_type': opt['option_type'],
                    'strike': opt['strike'],
                    'expiry_date': opt['expiry_date'],
                    'quantity': self.position_size,
                    'price': execution_result['executed_price'],
                    'action': index_action,
                    'delta': opt['delta'],
                    'implied_volatility': opt['implied_volatility'],
                    'entry_date': market_data['timestamp']
                })
                
                self.logger.info(f"{index_action} option d'indice: {opt['symbol']}")
        
        # Exécuter les ordres sur les composants
        for component in components_data:
            component_options = self.find_optimal_options(component['options_chain'], is_index=False)
            
            if not component_options:
                continue
                
            # Limiter à 1-2 options par composant pour diversifier
            for opt in component_options[:2]:
                # Ajuster la quantité en fonction du poids du composant
                adjusted_quantity = int(self.position_size * component['weight'] * 2)
                if adjusted_quantity < 1:
                    adjusted_quantity = 1
                
                execution_result = execution_handler.place_order(
                    symbol=opt['symbol'],
                    order_type=component_action,
                    quantity=adjusted_quantity,
                    price=opt['price']
                )
                
                if execution_result['success']:
                    self.positions['component_options'].append({
                        'symbol': opt['symbol'],
                        'component': component['symbol'],
                        'option_type': opt['option_type'],
                        'strike': opt['strike'],
                        'expiry_date': opt['expiry_date'],
                        'quantity': adjusted_quantity,
                        'price': execution_result['executed_price'],
                        'action': component_action,
                        'delta': opt['delta'],
                        'implied_volatility': opt['implied_volatility'],
                        'entry_date': market_data['timestamp'],
                        'weight': component['weight']
                    })
                    
                    self.logger.info(f"{component_action} option composant: {opt['symbol']} ({component['symbol']})")
        
        self.active_trade = True
        
        return {
            'success': True,
            'action': 'open_dispersion_trade',
            'correlation': implied_correlation,
            'index_options_count': len(self.positions['index_options']),
            'component_options_count': len(self.positions['component_options'])
        }
    
    def should_close_position(self, market_data):
        """
        Détermine s'il faut clôturer la position de dispersion actuelle.
        
        Args:
            market_data (dict): Données de marché actuelles
            
        Returns:
            bool: True si la position doit être clôturée
        """
        if not self.active_trade:
            return False
            
        # Recalculer la corrélation implicite actuelle
        index_data = market_data['index']
        components_data = market_data['components']
        
        index_iv = index_data['implied_volatility']
        component_ivs = [comp['implied_volatility'] for comp in components_data]
        component_weights = [comp['weight'] for comp in components_data]
        
        current_correlation = self.calculate_implied_correlation(
            index_iv, component_ivs, component_weights
        )
        
        # Clôturer si la corrélation dépasse le seuil
        if current_correlation > self.correlation_threshold + 0.1:
            self.logger.info(f"Signaux de clôture: corrélation {current_correlation:.4f} > seuil {self.correlation_threshold + 0.1:.4f}")
            return True
            
        # Vérifier le profit de la position
        current_pnl = self.calculate_pnl(market_data)
        
        # Clôturer si profit > 20% ou perte > 10%
        initial_investment = sum([opt['price'] * opt['quantity'] for opt in self.positions['index_options']]) + \
                            sum([opt['price'] * opt['quantity'] for opt in self.positions['component_options']])
        
        pnl_pct = current_pnl / initial_investment if initial_investment > 0 else 0
        
        if pnl_pct > 0.2:
            self.logger.info(f"Signaux de clôture: profit {pnl_pct:.2%} > 20%")
            return True
            
        if pnl_pct < -0.1:
            self.logger.info(f"Signaux de clôture: perte {pnl_pct:.2%} < -10%")
            return True
            
        # Vérifier le temps écoulé depuis l'entrée
        oldest_entry = min(
            [opt['entry_date'] for opt in self.positions['index_options']] + 
            [opt['entry_date'] for opt in self.positions['component_options']]
        )
        days_elapsed = (market_data['timestamp'] - oldest_entry).days
        
        # Clôturer si la position est ouverte depuis plus de 30 jours
        if days_elapsed > 30:
            self.logger.info(f"Signaux de clôture: position ouverte depuis {days_elapsed} jours > 30 jours")
            return True
            
        return False
    
    def close_position(self, market_data, execution_handler):
        """
        Clôture la position de dispersion actuelle.
        
        Args:
            market_data (dict): Données de marché actuelles
            execution_handler: Gestionnaire d'exécution des ordres
            
        Returns:
            dict: Résultat de la clôture
        """
        if not self.active_trade:
            return {'success': False, 'message': 'Pas de position active à clôturer'}
            
        # Clôturer les positions sur l'indice
        for opt in self.positions['index_options']:
            # L'action opposée à celle d'origine
            close_action = 'BUY' if opt['action'] == 'SELL' else 'SELL'
            
            execution_result = execution_handler.place_order(
                symbol=opt['symbol'],
                order_type=close_action,
                quantity=opt['quantity'],
                price=market_data['index']['options_prices'].get(opt['symbol'], opt['price'])
            )
            
            if execution_result['success']:
                self.logger.info(f"Clôture option d'indice: {close_action} {opt['symbol']}")
        
        # Clôturer les positions sur les composants
        for opt in self.positions['component_options']:
            close_action = 'SELL' if opt['action'] == 'BUY' else 'BUY'
            
            # Trouver le prix actuel dans les données de marché
            component_data = next((c for c in market_data['components'] if c['symbol'] == opt['component']), None)
            current_price = None
            
            if component_data:
                current_price = component_data['options_prices'].get(opt['symbol'], opt['price'])
            else:
                current_price = opt['price']
            
            execution_result = execution_handler.place_order(
                symbol=opt['symbol'],
                order_type=close_action,
                quantity=opt['quantity'],
                price=current_price
            )
            
            if execution_result['success']:
                self.logger.info(f"Clôture option composant: {close_action} {opt['symbol']} ({opt['component']})")
        
        # Calculer le PnL final
        final_pnl = self.calculate_pnl(market_data)
        
        # Réinitialiser les positions
        self.positions = {
            'index_options': [],
            'component_options': [],
        }
        self.active_trade = False
        
        return {
            'success': True,
            'action': 'close_dispersion_trade',
            'pnl': final_pnl
        }
    
    def calculate_pnl(self, market_data):
        """
        Calcule le PnL (Profit and Loss) de la stratégie de dispersion.
        
        Args:
            market_data (dict): Données de marché actuelles
            
        Returns:
            float: PnL actuel
        """
        if not self.active_trade:
            return 0.0
            
        total_pnl = 0.0
        
        # PnL des options d'indice
        index_prices = market_data['index']['options_prices']
        for opt in self.positions['index_options']:
            current_price = index_prices.get(opt['symbol'], opt['price'])
            
            if opt['action'] == 'SELL':
                # Pour une vente, profit = prix de vente - prix actuel
                opt_pnl = (opt['price'] - current_price) * opt['quantity']
            else:
                # Pour un achat, profit = prix actuel - prix d'achat
                opt_pnl = (current_price - opt['price']) * opt['quantity']
                
            total_pnl += opt_pnl
        
        # PnL des options de composants
        for opt in self.positions['component_options']:
            component_data = next((c for c in market_data['components'] if c['symbol'] == opt['component']), None)
            
            if not component_data:
                continue
                
            component_prices = component_data['options_prices']
            current_price = component_prices.get(opt['symbol'], opt['price'])
            
            if opt['action'] == 'BUY':
                opt_pnl = (current_price - opt['price']) * opt['quantity']
            else:
                opt_pnl = (opt['price'] - current_price) * opt['quantity']
                
            total_pnl += opt_pnl
        
        return total_pnl
    
    def run(self, market_data, execution_handler):
        """
        Exécute la stratégie de dispersion trading sur les données de marché actuelles.
        
        Args:
            market_data (dict): Données de marché actuelles
            execution_handler: Gestionnaire d'exécution des ordres
            
        Returns:
            dict: Résultat de l'exécution de la stratégie
        """
        # Vérifier si nous devons clôturer une position existante
        if self.active_trade and self.should_close_position(market_data):
            return self.close_position(market_data, execution_handler)
            
        # Chercher une nouvelle opportunité
        dispersion_opportunity, implied_correlation = self.is_dispersion_opportunity(market_data)
        
        if dispersion_opportunity and not self.active_trade:
            return self.execute_dispersion_trade(market_data, execution_handler, implied_correlation)
            
        # Pas d'action requise
        return {
            'success': True,
            'action': 'none',
            'message': 'Pas d\'opportunité de dispersion détectée' if not self.active_trade else 'Position active maintenue'
        }
