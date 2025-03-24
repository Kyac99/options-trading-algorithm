"""
Module pour la gestion des données de marché
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import json
import os
import time

from src.models.options_pricing import BlackScholesModel, ImpliedVolatility


class MarketDataHandler:
    """
    Gestionnaire de données de marché pour les options et les actifs sous-jacents.
    Récupère, nettoie et structure les données pour les stratégies de trading.
    """
    
    def __init__(self, config, data_provider=None):
        """
        Initialise le gestionnaire de données de marché.
        
        Args:
            config (dict): Configuration pour l'accès aux données
            data_provider: API ou service de données externe (optionnel)
        """
        self.config = config
        self.data_provider = data_provider
        self.logger = logging.getLogger(__name__)
        
        # Charger la configuration des données
        self.provider_name = config.get('provider', 'simulation')
        self.api_key = config.get('api_key', None)
        self.data_frequency = config.get('data_frequency', '1min')
        self.data_dir = config.get('data_dir', './data')
        
        # Modèles pour le calcul des prix et grecs
        self.bs_model = BlackScholesModel()
        self.iv_calculator = ImpliedVolatility()
        
        # Cache de données
        self.price_cache = {}
        self.options_cache = {}
        self.historical_data = {}
        
        # Créer le répertoire de données s'il n'existe pas
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.logger.info(f"Gestionnaire de données de marché initialisé avec provider: {self.provider_name}")
    
    def get_option_chain(self, underlying_symbol, expiry_date=None):
        """
        Récupère la chaîne complète d'options pour un sous-jacent donné.
        
        Args:
            underlying_symbol (str): Symbole de l'actif sous-jacent
            expiry_date (datetime, optional): Date d'expiration spécifique
            
        Returns:
            DataFrame: Chaîne d'options structurée
        """
        if self.data_provider is None:
            # Mode simulation - générer des données synthétiques
            return self._generate_synthetic_option_chain(underlying_symbol, expiry_date)
        else:
            # Récupérer les données via l'API de données
            return self._fetch_option_chain_from_provider(underlying_symbol, expiry_date)
    
    def get_historical_prices(self, symbol, start_date, end_date=None, frequency='1d'):
        """
        Récupère les prix historiques pour un symbole donné.
        
        Args:
            symbol (str): Symbole de l'instrument
            start_date (datetime): Date de début
            end_date (datetime, optional): Date de fin (par défaut: aujourd'hui)
            frequency (str): Fréquence des données ('1d', '1h', etc.)
            
        Returns:
            DataFrame: Prix historiques
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Vérifier si on a déjà les données en cache
        cache_key = f"{symbol}_{frequency}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        if cache_key in self.historical_data:
            return self.historical_data[cache_key]
        
        if self.data_provider is None:
            # Mode simulation - générer des données synthétiques
            historical_data = self._generate_synthetic_price_history(symbol, start_date, end_date, frequency)
        else:
            # Récupérer les données via l'API de données
            historical_data = self._fetch_historical_data_from_provider(symbol, start_date, end_date, frequency)
        
        # Mettre en cache
        self.historical_data[cache_key] = historical_data
        
        return historical_data
    
    def get_current_price(self, symbol):
        """
        Récupère le prix actuel d'un instrument.
        
        Args:
            symbol (str): Symbole de l'instrument
            
        Returns:
            float: Prix actuel
        """
        # Vérifier si on a un prix en cache qui est récent (moins de 5 secondes)
        current_time = datetime.now()
        if symbol in self.price_cache:
            cache_time, price = self.price_cache[symbol]
            if (current_time - cache_time).total_seconds() < 5:
                return price
        
        if self.data_provider is None:
            # Mode simulation - générer un prix aléatoire
            last_price = self._generate_synthetic_price(symbol)
        else:
            # Récupérer le prix via l'API de données
            last_price = self._fetch_current_price_from_provider(symbol)
        
        # Mettre en cache
        self.price_cache[symbol] = (current_time, last_price)
        
        return last_price
    
    def get_volatility_surface(self, underlying_symbol):
        """
        Construit une surface de volatilité à partir des options disponibles.
        
        Args:
            underlying_symbol (str): Symbole de l'actif sous-jacent
            
        Returns:
            DataFrame: Surface de volatilité (expiration x strike)
        """
        # Récupérer toutes les options du sous-jacent
        options_chain = self.get_option_chain(underlying_symbol)
        
        if options_chain.empty:
            self.logger.warning(f"Pas de données d'options pour construire la surface de volatilité: {underlying_symbol}")
            return pd.DataFrame()
        
        # Extraire les informations uniques
        expirations = sorted(options_chain['expiry_date'].unique())
        strikes = sorted(options_chain['strike'].unique())
        
        # Initialiser une matrice pour la surface de volatilité
        vol_matrix = np.zeros((len(expirations), len(strikes)))
        
        # Remplir la matrice avec les volatilités implicites
        for i, expiry in enumerate(expirations):
            for j, strike in enumerate(strikes):
                # Filtrer les options correspondantes
                filtered = options_chain[
                    (options_chain['expiry_date'] == expiry) & 
                    (options_chain['strike'] == strike)
                ]
                
                if not filtered.empty:
                    # Faire la moyenne des volatilités des calls et puts
                    vol_matrix[i, j] = filtered['implied_volatility'].mean()
        
        # Convertir en DataFrame pour une manipulation plus facile
        vol_surface = pd.DataFrame(vol_matrix, index=expirations, columns=strikes)
        
        return vol_surface
    
    def calculate_greeks(self, options_data, underlying_price=None, risk_free_rate=0.02):
        """
        Calcule les Grecques pour les options données.
        
        Args:
            options_data (DataFrame): Données d'options
            underlying_price (float, optional): Prix actuel du sous-jacent
            risk_free_rate (float): Taux sans risque
            
        Returns:
            DataFrame: Options avec les Grecques calculées
        """
        if options_data.empty:
            return options_data
        
        # Copier pour ne pas modifier l'original
        result = options_data.copy()
        
        # Si le prix du sous-jacent n'est pas fourni, utiliser la première valeur des données
        if underlying_price is None and 'underlying_price' in result.columns:
            underlying_price = result['underlying_price'].iloc[0]
        elif underlying_price is None:
            self.logger.error("Prix du sous-jacent requis pour calculer les Grecques")
            return result
        
        # Calculer les Grecques pour chaque option
        for idx, row in result.iterrows():
            time_to_expiry = (row['expiry_date'] - datetime.now()).total_seconds() / (365 * 24 * 60 * 60)
            
            if time_to_expiry <= 0:
                # Option expirée, pas de calcul de Grecques
                continue
            
            # Récupérer les paramètres nécessaires
            strike = row['strike']
            option_type = row['option_type']
            implied_vol = row['implied_volatility']
            
            # Calculer les Grecques selon le type d'option
            if option_type == 'call':
                result.at[idx, 'delta'] = self.bs_model.delta_call(
                    underlying_price, strike, time_to_expiry, risk_free_rate, implied_vol
                )
                result.at[idx, 'theta'] = self.bs_model.theta_call(
                    underlying_price, strike, time_to_expiry, risk_free_rate, implied_vol
                )
            else:  # put
                result.at[idx, 'delta'] = self.bs_model.delta_put(
                    underlying_price, strike, time_to_expiry, risk_free_rate, implied_vol
                )
                result.at[idx, 'theta'] = self.bs_model.theta_put(
                    underlying_price, strike, time_to_expiry, risk_free_rate, implied_vol
                )
            
            # Grecques communes pour calls et puts
            result.at[idx, 'gamma'] = self.bs_model.gamma(
                underlying_price, strike, time_to_expiry, risk_free_rate, implied_vol
            )
            result.at[idx, 'vega'] = self.bs_model.vega(
                underlying_price, strike, time_to_expiry, risk_free_rate, implied_vol
            )
        
        return result
    
    def calculate_implied_volatility(self, options_data, underlying_price=None, risk_free_rate=0.02):
        """
        Calcule la volatilité implicite pour les options données.
        
        Args:
            options_data (DataFrame): Données d'options
            underlying_price (float, optional): Prix actuel du sous-jacent
            risk_free_rate (float): Taux sans risque
            
        Returns:
            DataFrame: Options avec volatilité implicite calculée
        """
        if options_data.empty:
            return options_data
        
        # Copier pour ne pas modifier l'original
        result = options_data.copy()
        
        # Si le prix du sous-jacent n'est pas fourni, utiliser la première valeur des données
        if underlying_price is None and 'underlying_price' in result.columns:
            underlying_price = result['underlying_price'].iloc[0]
        elif underlying_price is None:
            self.logger.error("Prix du sous-jacent requis pour calculer la volatilité implicite")
            return result
        
        # Calculer la volatilité implicite pour chaque option
        for idx, row in result.iterrows():
            time_to_expiry = (row['expiry_date'] - datetime.now()).total_seconds() / (365 * 24 * 60 * 60)
            
            if time_to_expiry <= 0:
                # Option expirée, pas de calcul de volatilité implicite
                continue
            
            # Récupérer les paramètres nécessaires
            market_price = row['price']
            strike = row['strike']
            option_type = row['option_type']
            
            # Calculer la volatilité implicite selon le type d'option
            if option_type == 'call':
                iv = self.iv_calculator.calculate_call(
                    market_price, underlying_price, strike, time_to_expiry, risk_free_rate
                )
            else:  # put
                iv = self.iv_calculator.calculate_put(
                    market_price, underlying_price, strike, time_to_expiry, risk_free_rate
                )
            
            result.at[idx, 'implied_volatility'] = iv
        
        return result
    
    def _generate_synthetic_option_chain(self, underlying_symbol, expiry_date=None):
        """
        Génère une chaîne d'options synthétique pour les tests et simulations.
        
        Args:
            underlying_symbol (str): Symbole de l'actif sous-jacent
            expiry_date (datetime, optional): Date d'expiration spécifique
            
        Returns:
            DataFrame: Chaîne d'options synthétique
        """
        self.logger.info(f"Génération d'une chaîne d'options synthétique pour {underlying_symbol}")
        
        # Obtenir ou générer le prix du sous-jacent
        underlying_price = self.get_current_price(underlying_symbol)
        
        # Si aucune date d'expiration n'est spécifiée, générer plusieurs expirations
        current_date = datetime.now()
        
        if expiry_date is None:
            expiry_dates = [
                current_date + timedelta(days=30),  # 1 mois
                current_date + timedelta(days=60),  # 2 mois
                current_date + timedelta(days=90),  # 3 mois
                current_date + timedelta(days=180)  # 6 mois
            ]
        else:
            expiry_dates = [expiry_date]
        
        # Générer une gamme de strikes autour du prix actuel
        strike_range = 0.3  # +/- 30%
        min_strike = underlying_price * (1 - strike_range)
        max_strike = underlying_price * (1 + strike_range)
        
        strikes = np.linspace(min_strike, max_strike, 15)
        strikes = [round(strike, 2) for strike in strikes]
        
        # Créer la structure de données
        option_data = []
        
        # ATM volatilité de base
        base_volatility = 0.25
        
        for expiry in expiry_dates:
            time_to_expiry = (expiry - current_date).days / 365.0
            
            # Volatilité augmente avec l'expiration
            expiry_vol_factor = 1 + (time_to_expiry - 30/365) * 0.5
            expiry_volatility = base_volatility * max(0.8, min(1.5, expiry_vol_factor))
            
            for strike in strikes:
                # Créer un skew de volatilité (plus élevé pour les puts OTM, plus bas pour les calls OTM)
                moneyness = strike / underlying_price
                # Fonction de skew: plus élevée pour les bas strikes, plus basse pour les hauts strikes
                vol_skew = 0.1 * (1 - moneyness) 
                
                # Volatilité avec skew
                call_iv = expiry_volatility - vol_skew
                put_iv = expiry_volatility + vol_skew
                
                # Limiter la volatilité à des valeurs raisonnables
                call_iv = max(0.1, min(0.8, call_iv))
                put_iv = max(0.1, min(0.8, put_iv))
                
                # Prix des options avec Black-Scholes
                call_price = self.bs_model.call_price(
                    underlying_price, strike, time_to_expiry, 0.02, call_iv
                )
                put_price = self.bs_model.put_price(
                    underlying_price, strike, time_to_expiry, 0.02, put_iv
                )
                
                # Arrondir les prix
                call_price = round(max(0.01, call_price), 2)
                put_price = round(max(0.01, put_price), 2)
                
                # Ajouter le call
                call_data = {
                    'symbol': f"{underlying_symbol}C{expiry.strftime('%y%m%d')}{int(strike*100)}",
                    'underlying_symbol': underlying_symbol,
                    'underlying_price': underlying_price,
                    'option_type': 'call',
                    'strike': strike,
                    'expiry_date': expiry,
                    'price': call_price,
                    'bid': call_price * 0.98,  # Simuler un spread
                    'ask': call_price * 1.02,
                    'mid_price': call_price,
                    'volume': np.random.randint(10, 1000),
                    'open_interest': np.random.randint(100, 5000),
                    'implied_volatility': call_iv,
                    'delta': None,
                    'gamma': None,
                    'theta': None,
                    'vega': None
                }
                option_data.append(call_data)
                
                # Ajouter le put
                put_data = {
                    'symbol': f"{underlying_symbol}P{expiry.strftime('%y%m%d')}{int(strike*100)}",
                    'underlying_symbol': underlying_symbol,
                    'underlying_price': underlying_price,
                    'option_type': 'put',
                    'strike': strike,
                    'expiry_date': expiry,
                    'price': put_price,
                    'bid': put_price * 0.98,
                    'ask': put_price * 1.02,
                    'mid_price': put_price,
                    'volume': np.random.randint(10, 1000),
                    'open_interest': np.random.randint(100, 5000),
                    'implied_volatility': put_iv,
                    'delta': None,
                    'gamma': None,
                    'theta': None,
                    'vega': None
                }
                option_data.append(put_data)
        
        # Convertir en DataFrame
        df = pd.DataFrame(option_data)
        
        # Calculer les grecques
        df = self.calculate_greeks(df, underlying_price, 0.02)
        
        return df
    
    def _simulate_index_components(self, index_symbol, num_components=10):
        """
        Simule les composants d'un indice pour la stratégie de dispersion.
        
        Args:
            index_symbol (str): Symbole de l'indice
            num_components (int): Nombre de composants à simuler
            
        Returns:
            list: Liste des composants avec leurs données
        """
        components = []
        
        # Simuler une volatilité de base pour l'indice
        index_vol = 0.25
        
        for i in range(num_components):
            component_symbol = f"{index_symbol}_{i+1}"
            component_weight = round(1 / num_components, 3)
            
            # Simuler un prix qui varie autour du prix de l'indice
            component_price = self.get_current_price(component_symbol)
            
            # Simuler une chaîne d'options pour ce composant
            component_options = self._generate_synthetic_option_chain(component_symbol)
            
            # Calculer une volatilité implicite moyenne pour ce composant
            # Les composants ont généralement une volatilité plus élevée que l'indice
            component_iv = index_vol * (1 + 0.2 * np.random.random())
            
            # Créer un dictionnaire pour ce composant
            component = {
                'symbol': component_symbol,
                'weight': component_weight,
                'price': component_price,
                'options_chain': component_options,
                'implied_volatility': component_iv,
                'options_prices': {row['symbol']: row['price'] for _, row in component_options.iterrows()}
            }
            
            components.append(component)
        
        return components
    
    def _generate_synthetic_price_history(self, symbol, start_date, end_date, frequency='1d'):
        """
        Génère des prix historiques synthétiques pour les tests et simulations.
        """
        # Implementation as previously defined
        # (Rest of the method remains the same)
        
    def save_to_csv(self, data, filename):
        """
        Sauvegarde les données dans un fichier CSV.
        
        Args:
            data (DataFrame): Données à sauvegarder
            filename (str): Nom du fichier
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        file_path = os.path.join(self.data_dir, filename)
        data.to_csv(file_path)
        self.logger.info(f"Données sauvegardées dans {file_path}")
        return file_path
    
    def load_from_csv(self, filename):
        """
        Charge les données depuis un fichier CSV.
        
        Args:
            filename (str): Nom du fichier
            
        Returns:
            DataFrame: Données chargées
        """
        file_path = os.path.join(self.data_dir, filename)
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            self.logger.info(f"Données chargées depuis {file_path}")
            return data
        else:
            self.logger.warning(f"Fichier {file_path} non trouvé")
            return pd.DataFrame()
