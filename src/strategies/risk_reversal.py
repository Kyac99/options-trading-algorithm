"""
Module pour la stratégie de Risk Reversal
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from src.models.options_pricing import BlackScholesModel


class RiskReversalStrategy:
    """
    Stratégie de Risk Reversal - Exploite l'asymétrie du skew de volatilité en vendant 
    des options out-of-the-money (OTM) d'un côté et en achetant des options OTM de l'autre côté.
    Typiquement, vendre des puts OTM et acheter des calls OTM pour une position synthétique longue,
    ou vendre des calls OTM et acheter des puts OTM pour une position synthétique courte.
    """
    
    def __init__(self, config):
        """
        Initialise la stratégie de Risk Reversal.
        
        Args:
            config (dict): Configuration de la stratégie
        """
        self.config = config
        self.bs_model = BlackScholesModel()
        self.logger = logging.getLogger(__name__)
        
        # Paramètres de la stratégie
        self.skew_threshold = config.get('skew_threshold', 0.15)
        self.expiry_range = config.get('expiry_range', [30, 90])
        self.position_size = config.get('position_size', 75)
        self.otm_delta_target = config.get('otm_delta_target', 0.25)  # Delta cible pour les options OTM
        
        # État de la stratégie
        self.positions = {
            'short_options': [],  # Options vendues (généralement puts OTM)
            'long_options': [],   # Options achetées (généralement calls OTM)
        }
        self.skew_history = []
        self.active_trade = False
        self.trade_direction = None  # 'bullish' ou 'bearish'
        
        self.logger.info("Stratégie de Risk Reversal initialisée")
    
    def calculate_volatility_skew(self, options_chain):
        """
        Calcule le skew de volatilité entre les calls et les puts OTM.
        
        Args:
            options_chain (DataFrame): Chaîne d'options disponibles
            
        Returns:
            float: Mesure du skew de volatilité (positif indique que les puts OTM ont une vol plus élevée)
        """
        # Filtrer les options par type
        calls = options_chain[options_chain['option_type'] == 'call']
        puts = options_chain[options_chain['option_type'] == 'put']
        
        if calls.empty or puts.empty:
            self.logger.warning("Données insuffisantes pour calculer le skew")
            return 0
        
        # Filtrer les options OTM
        otm_calls = calls[calls['delta'] < 0.5]
        otm_puts = puts[puts['delta'] > -0.5]
        
        if otm_calls.empty or otm_puts.empty:
            self.logger.warning("Options OTM insuffisantes pour calculer le skew")
            return 0
        
        # Sélectionner les options avec delta proche de +/- otm_delta_target
        target_calls = otm_calls.iloc[(otm_calls['delta'] - self.otm_delta_target).abs().argsort()[:3]]
        target_puts = otm_puts.iloc[(otm_puts['delta'] + self.otm_delta_target).abs().argsort()[:3]]
        
        # Calculer la volatilité implicite moyenne pour chaque groupe
        avg_call_iv = target_calls['implied_volatility'].mean()
        avg_put_iv = target_puts['implied_volatility'].mean()
        
        # Calculer le skew (différence entre vol des puts et des calls)
        skew = avg_put_iv - avg_call_iv
        
        self.logger.info(f"Skew de volatilité calculé: {skew:.4f} (Put IV: {avg_put_iv:.4f}, Call IV: {avg_call_iv:.4f})")
        return skew
    
    def is_skew_opportunity(self, market_data):
        """
        Détermine s'il existe une opportunité de trading basée sur le skew de volatilité.
        
        Args:
            market_data (dict): Données de marché actuelles
            
        Returns:
            tuple: (bool, str) - Opportunité détectée et direction ('bullish' ou 'bearish')
        """
        options_chain = market_data['options_chain']
        
        # Calculer le skew de volatilité
        skew = self.calculate_volatility_skew(options_chain)
        
        # Enregistrer l'historique de skew
        self.skew_history.append({
            'timestamp': market_data['timestamp'],
            'skew': skew,
            'underlying_price': market_data['underlying_price']
        })
        
        # Déterminer la direction du trade basée sur le skew
        if skew > self.skew_threshold:
            # Skew positif excessif (puts OTM trop chers) => stratégie bullish
            # Vendre des puts OTM, acheter des calls OTM
            direction = 'bullish'
            opportunity = True
        elif skew < -self.skew_threshold:
            # Skew négatif excessif (calls OTM trop chers) => stratégie bearish
            # Vendre des calls OTM, acheter des puts OTM
            direction = 'bearish'
            opportunity = True
        else:
            # Skew dans la plage normale, pas d'opportunité
            direction = None
            opportunity = False
        
        if opportunity:
            self.logger.info(f"Opportunité de Risk Reversal détectée: {direction}, skew: {skew:.4f}")
        else:
            self.logger.debug(f"Pas d'opportunité de Risk Reversal, skew: {skew:.4f}")
            
        return opportunity, direction
    
    def find_otm_options(self, options_chain, direction, expiry_min, expiry_max):
        """
        Sélectionne les options OTM appropriées pour la stratégie de Risk Reversal.
        
        Args:
            options_chain (DataFrame): Chaîne d'options disponibles
            direction (str): Direction de la stratégie ('bullish' ou 'bearish')
            expiry_min (datetime): Date d'expiration minimale
            expiry_max (datetime): Date d'expiration maximale
            
        Returns:
            tuple: (short_options, long_options) - Options à vendre et à acheter
        """
        # Filtrer les options par expiration
        filtered_options = options_chain[
            (options_chain['expiry_date'] >= expiry_min) &
            (options_chain['expiry_date'] <= expiry_max)
        ]
        
        if filtered_options.empty:
            self.logger.warning("Aucune option avec l'expiration cible trouvée")
            return [], []
        
        # Séparer les calls et puts
        calls = filtered_options[filtered_options['option_type'] == 'call']
        puts = filtered_options[filtered_options['option_type'] == 'put']
        
        if calls.empty or puts.empty:
            self.logger.warning("Données d'options insuffisantes")
            return [], []
        
        # Options à vendre et à acheter selon la direction
        if direction == 'bullish':
            # Pour une stratégie bullish, vendre des puts OTM et acheter des calls OTM
            short_options_df = puts[puts['delta'] > -0.5]  # Puts OTM
            long_options_df = calls[calls['delta'] < 0.5]   # Calls OTM
            
            # Filtrer par delta cible
            short_options_df = short_options_df.iloc[
                (short_options_df['delta'] + self.otm_delta_target).abs().argsort()
            ]
            long_options_df = long_options_df.iloc[
                (long_options_df['delta'] - self.otm_delta_target).abs().argsort()
            ]
        else:  # bearish
            # Pour une stratégie bearish, vendre des calls OTM et acheter des puts OTM
            short_options_df = calls[calls['delta'] < 0.5]  # Calls OTM
            long_options_df = puts[puts['delta'] > -0.5]    # Puts OTM
            
            # Filtrer par delta cible
            short_options_df = short_options_df.iloc[
                (short_options_df['delta'] - self.otm_delta_target).abs().argsort()
            ]
            long_options_df = long_options_df.iloc[
                (long_options_df['delta'] + self.otm_delta_target).abs().argsort()
            ]
        
        # Sélectionner les meilleures options (liquidité, spread)
        if not short_options_df.empty and not long_options_df.empty:
            short_options_df['spread_pct'] = (short_options_df['ask'] - short_options_df['bid']) / short_options_df['mid_price']
            long_options_df['spread_pct'] = (long_options_df['ask'] - long_options_df['bid']) / long_options_df['mid_price']
            
            short_options_df = short_options_df.sort_values('spread_pct')
            long_options_df = long_options_df.sort_values('spread_pct')
            
            short_options = short_options_df.head(3).to_dict('records')
            long_options = long_options_df.head(3).to_dict('records')
        else:
            short_options = []
            long_options = []
        
        self.logger.info(f"Options OTM trouvées: {len(short_options)} à vendre, {len(long_options)} à acheter")
        return short_options, long_options
    
    def execute_risk_reversal(self, market_data, execution_handler, direction):
        """
        Exécute un trade de Risk Reversal.
        
        Args:
            market_data (dict): Données de marché actuelles
            execution_handler: Gestionnaire d'exécution des ordres
            direction (str): Direction du trade ('bullish' ou 'bearish')
            
        Returns:
            dict: Résultat de l'exécution du trade
        """
        # Si un trade est déjà actif, ne pas en ouvrir un nouveau
        if self.active_trade:
            self.logger.info("Trade de Risk Reversal déjà actif, pas de nouveau trade ouvert")
            return {'success': False, 'message': 'Trade déjà actif'}
        
        options_chain = market_data['options_chain']
        current_time = market_data['timestamp']
        
        # Déterminer les dates d'expiration cibles
        expiry_min = current_time + timedelta(days=self.expiry_range[0])
        expiry_max = current_time + timedelta(days=self.expiry_range[1])
        
        # Trouver les options OTM appropriées
        short_options, long_options = self.find_otm_options(
            options_chain, direction, expiry_min, expiry_max
        )
        
        if not short_options or not long_options:
            return {'success': False, 'message': 'Options appropriées non trouvées'}
        
        # Exécuter les ordres pour les options vendues (short)
        for opt in short_options:
            execution_result = execution_handler.place_order(
                symbol=opt['symbol'],
                order_type='SELL',
                quantity=self.position_size,
                price=opt['price']
            )
            
            if execution_result['success']:
                self.positions['short_options'].append({
                    'symbol': opt['symbol'],
                    'option_type': opt['option_type'],
                    'strike': opt['strike'],
                    'expiry_date': opt['expiry_date'],
                    'quantity': self.position_size,
                    'price': execution_result['executed_price'],
                    'delta': opt['delta'],
                    'implied_volatility': opt['implied_volatility'],
                    'entry_date': current_time
                })
                
                self.logger.info(f"Vente option {opt['option_type']}: {opt['symbol']}")
        
        # Exécuter les ordres pour les options achetées (long)
        for opt in long_options:
            execution_result = execution_handler.place_order(
                symbol=opt['symbol'],
                order_type='BUY',
                quantity=self.position_size,
                price=opt['price']
            )
            
            if execution_result['success']:
                self.positions['long_options'].append({
                    'symbol': opt['symbol'],
                    'option_type': opt['option_type'],
                    'strike': opt['strike'],
                    'expiry_date': opt['expiry_date'],
                    'quantity': self.position_size,
                    'price': execution_result['executed_price'],
                    'delta': opt['delta'],
                    'implied_volatility': opt['implied_volatility'],
                    'entry_date': current_time
                })
                
                self.logger.info(f"Achat option {opt['option_type']}: {opt['symbol']}")
        
        self.active_trade = True
        self.trade_direction = direction
        
        return {
            'success': True,
            'action': 'open_risk_reversal',
            'direction': direction,
            'short_options_count': len(self.positions['short_options']),
            'long_options_count': len(self.positions['long_options'])
        }
    
    def should_close_position(self, market_data):
        """
        Détermine s'il faut clôturer la position de Risk Reversal actuelle.
        
        Args:
            market_data (dict): Données de marché actuelles
            
        Returns:
            bool: True si la position doit être clôturée
        """
        if not self.active_trade:
            return False
            
        # Recalculer le skew actuel
        current_skew = self.calculate_volatility_skew(market_data['options_chain'])
        
        # Clôturer si le skew est revenu à la normale ou s'est inversé
        if self.trade_direction == 'bullish' and current_skew < self.skew_threshold / 2:
            self.logger.info(f"Signaux de clôture: skew réduit {current_skew:.4f} < {self.skew_threshold/2:.4f}")
            return True
            
        if self.trade_direction == 'bearish' and current_skew > -self.skew_threshold / 2:
            self.logger.info(f"Signaux de clôture: skew réduit {current_skew:.4f} > {-self.skew_threshold/2:.4f}")
            return True
            
        # Vérifier le profit de la position
        current_pnl = self.calculate_pnl(market_data)
        
        # Calculer l'investissement initial
        initial_investment = sum([opt['price'] * opt['quantity'] for opt in self.positions['long_options']]) - \
                            sum([opt['price'] * opt['quantity'] for opt in self.positions['short_options']])
        
        # Éviter la division par zéro
        if abs(initial_investment) < 1e-6:
            pnl_pct = 0
        else:
            pnl_pct = current_pnl / abs(initial_investment)
        
        # Clôturer si profit > 50% ou perte > 20%
        if pnl_pct > 0.5:
            self.logger.info(f"Signaux de clôture: profit {pnl_pct:.2%} > 50%")
            return True
            
        if pnl_pct < -0.2:
            self.logger.info(f"Signaux de clôture: perte {pnl_pct:.2%} < -20%")
            return True
            
        # Vérifier si l'option la plus proche de l'expiration arrive à échéance bientôt
        earliest_expiry = min(
            [opt['expiry_date'] for opt in self.positions['short_options'] + self.positions['long_options']]
        )
        days_to_expiry = (earliest_expiry - market_data['timestamp']).days
        
        # Clôturer si moins de 14 jours avant expiration
        if days_to_expiry < 14:
            self.logger.info(f"Signaux de clôture: proche de l'expiration ({days_to_expiry} jours)")
            return True
            
        # Vérifier le delta de la position
        if self.trade_direction == 'bullish':
            # Pour une position bullish, on veut fermer si le delta devient trop négatif
            total_delta = self.calculate_position_delta(market_data)
            if total_delta < -10:
                self.logger.info(f"Signaux de clôture: delta négatif {total_delta} < -10 pour position bullish")
                return True
        else:  # bearish
            # Pour une position bearish, on veut fermer si le delta devient trop positif
            total_delta = self.calculate_position_delta(market_data)
            if total_delta > 10:
                self.logger.info(f"Signaux de clôture: delta positif {total_delta} > 10 pour position bearish")
                return True
                
        return False
    
    def close_position(self, market_data, execution_handler):
        """
        Clôture la position de Risk Reversal actuelle.
        
        Args:
            market_data (dict): Données de marché actuelles
            execution_handler: Gestionnaire d'exécution des ordres
            
        Returns:
            dict: Résultat de la clôture
        """
        if not self.active_trade:
            return {'success': False, 'message': 'Pas de position active à clôturer'}
            
        # Clôturer les positions short (rachat)
        for opt in self.positions['short_options']:
            execution_result = execution_handler.place_order(
                symbol=opt['symbol'],
                order_type='BUY',  # Opposé de l'origine SELL
                quantity=opt['quantity'],
                price=market_data['options_prices'].get(opt['symbol'], opt['price'])
            )
            
            if execution_result['success']:
                self.logger.info(f"Clôture option vendue: rachat {opt['symbol']}")
        
        # Clôturer les positions long (vente)
        for opt in self.positions['long_options']:
            execution_result = execution_handler.place_order(
                symbol=opt['symbol'],
                order_type='SELL',  # Opposé de l'origine BUY
                quantity=opt['quantity'],
                price=market_data['options_prices'].get(opt['symbol'], opt['price'])
            )
            
            if execution_result['success']:
                self.logger.info(f"Clôture option achetée: vente {opt['symbol']}")
        
        # Calculer le PnL final
        final_pnl = self.calculate_pnl(market_data)
        
        # Réinitialiser les positions
        self.positions = {
            'short_options': [],
            'long_options': [],
        }
        self.active_trade = False
        self.trade_direction = None
        
        return {
            'success': True,
            'action': 'close_risk_reversal',
            'pnl': final_pnl
        }
    
    def calculate_pnl(self, market_data):
        """
        Calcule le PnL (Profit and Loss) de la stratégie de Risk Reversal.
        
        Args:
            market_data (dict): Données de marché actuelles
            
        Returns:
            float: PnL actuel
        """
        if not self.active_trade:
            return 0.0
            
        total_pnl = 0.0
        options_prices = market_data['options_prices']
        
        # PnL des options short (vendues)
        for opt in self.positions['short_options']:
            current_price = options_prices.get(opt['symbol'], opt['price'])
            # Pour les options vendues, le PnL est positif si le prix actuel est inférieur au prix d'entrée
            opt_pnl = (opt['price'] - current_price) * opt['quantity']
            total_pnl += opt_pnl
        
        # PnL des options long (achetées)
        for opt in self.positions['long_options']:
            current_price = options_prices.get(opt['symbol'], opt['price'])
            # Pour les options achetées, le PnL est positif si le prix actuel est supérieur au prix d'entrée
            opt_pnl = (current_price - opt['price']) * opt['quantity']
            total_pnl += opt_pnl
        
        return total_pnl
    
    def calculate_position_delta(self, market_data):
        """
        Calcule le delta total de la position.
        
        Args:
            market_data (dict): Données de marché actuelles
            
        Returns:
            float: Delta total de la position
        """
        total_delta = 0.0
        underlying_price = market_data['underlying_price']
        risk_free_rate = market_data.get('risk_free_rate', 0.02)
        
        # Mettre à jour le delta de chaque option
        for opt_list in [self.positions['short_options'], self.positions['long_options']]:
            for opt in opt_list:
                time_to_expiry = (opt['expiry_date'] - market_data['timestamp']).days / 365.0
                
                if time_to_expiry <= 0:
                    continue
                    
                # Calculer le delta actuel
                if opt['option_type'] == 'call':
                    new_delta = self.bs_model.delta_call(
                        underlying_price, opt['strike'], time_to_expiry, 
                        risk_free_rate, opt['implied_volatility']
                    )
                else:  # put
                    new_delta = self.bs_model.delta_put(
                        underlying_price, opt['strike'], time_to_expiry, 
                        risk_free_rate, opt['implied_volatility']
                    )
                
                # Somme pondérée des deltas
                if opt in self.positions['short_options']:
                    # Pour les options vendues, le delta est inversé
                    total_delta -= new_delta * opt['quantity']
                else:
                    total_delta += new_delta * opt['quantity']
        
        return total_delta
    
    def run(self, market_data, execution_handler):
        """
        Exécute la stratégie de Risk Reversal sur les données de marché actuelles.
        
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
        skew_opportunity, direction = self.is_skew_opportunity(market_data)
        
        if skew_opportunity and not self.active_trade:
            return self.execute_risk_reversal(market_data, execution_handler, direction)
            
        # Pas d'action requise
        return {
            'success': True,
            'action': 'none',
            'message': 'Pas d\'opportunité de Risk Reversal détectée' if not self.active_trade else 'Position active maintenue'
        }
