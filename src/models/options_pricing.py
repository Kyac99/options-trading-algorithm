"""
Module de pricing d'options et de calcul des Grecques
"""
import numpy as np
from scipy.stats import norm
import pandas as pd


class BlackScholesModel:
    """
    Implémentation du modèle Black-Scholes pour le pricing d'options et le calcul des grecques.
    """

    @staticmethod
    def d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le paramètre d1 de la formule de Black-Scholes.
        
        Args:
            spot (float): Prix du sous-jacent
            strike (float): Prix d'exercice de l'option
            time_to_expiry (float): Temps jusqu'à l'expiration en années
            risk_free_rate (float): Taux sans risque (décimal, ex: 0.05 pour 5%)
            volatility (float): Volatilité implicite (décimal, ex: 0.2 pour 20%)
            dividend_yield (float, optional): Rendement du dividende. Par défaut 0.
            
        Returns:
            float: Valeur de d1
        """
        if time_to_expiry <= 0 or volatility <= 0:
            return float('nan')
        
        return (np.log(spot / strike) + 
                (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
                (volatility * np.sqrt(time_to_expiry))

    @staticmethod
    def d2(d1_value, volatility, time_to_expiry):
        """
        Calcule le paramètre d2 de la formule de Black-Scholes.
        
        Args:
            d1_value (float): Valeur de d1
            volatility (float): Volatilité implicite
            time_to_expiry (float): Temps jusqu'à l'expiration en années
            
        Returns:
            float: Valeur de d2
        """
        return d1_value - volatility * np.sqrt(time_to_expiry)

    def call_price(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le prix d'une option d'achat (call) selon le modèle Black-Scholes.
        
        Args:
            spot (float): Prix du sous-jacent
            strike (float): Prix d'exercice de l'option
            time_to_expiry (float): Temps jusqu'à l'expiration en années
            risk_free_rate (float): Taux sans risque (décimal)
            volatility (float): Volatilité implicite (décimal)
            dividend_yield (float, optional): Rendement du dividende. Par défaut 0.
            
        Returns:
            float: Prix de l'option d'achat
        """
        if time_to_expiry <= 0:
            return max(0, spot - strike)
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        d2_val = self.d2(d1_val, volatility, time_to_expiry)
        
        return spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1_val) - \
               strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2_val)

    def put_price(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le prix d'une option de vente (put) selon le modèle Black-Scholes.
        
        Args:
            spot (float): Prix du sous-jacent
            strike (float): Prix d'exercice de l'option
            time_to_expiry (float): Temps jusqu'à l'expiration en années
            risk_free_rate (float): Taux sans risque (décimal)
            volatility (float): Volatilité implicite (décimal)
            dividend_yield (float, optional): Rendement du dividende. Par défaut 0.
            
        Returns:
            float: Prix de l'option de vente
        """
        if time_to_expiry <= 0:
            return max(0, strike - spot)
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        d2_val = self.d2(d1_val, volatility, time_to_expiry)
        
        return strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2_val) - \
               spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1_val)

    def delta_call(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le Delta d'un call (sensibilité du prix par rapport au sous-jacent).
        
        Returns:
            float: Delta de l'option call
        """
        if time_to_expiry <= 0:
            return 1.0 if spot > strike else 0.0
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        return np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1_val)

    def delta_put(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le Delta d'un put.
        
        Returns:
            float: Delta de l'option put
        """
        if time_to_expiry <= 0:
            return -1.0 if spot < strike else 0.0
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        return np.exp(-dividend_yield * time_to_expiry) * (norm.cdf(d1_val) - 1)

    def gamma(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le Gamma (sensibilité du Delta par rapport au sous-jacent).
        Le Gamma est identique pour les calls et les puts.
        
        Returns:
            float: Gamma de l'option
        """
        if time_to_expiry <= 0:
            return 0.0
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        
        return np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1_val) / \
               (spot * volatility * np.sqrt(time_to_expiry))

    def vega(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le Vega (sensibilité du prix par rapport à la volatilité).
        Le Vega est identique pour les calls et les puts.
        
        Returns:
            float: Vega de l'option (pour un changement de volatilité de 1%)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        
        # Vega pour un changement de 1% de volatilité (0.01)
        return 0.01 * spot * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1_val) * np.sqrt(time_to_expiry)

    def theta_call(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le Theta d'un call (sensibilité du prix par rapport au temps).
        
        Returns:
            float: Theta de l'option call (pour un jour)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        d2_val = self.d2(d1_val, volatility, time_to_expiry)
        
        theta = -spot * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1_val) * volatility / \
                (2 * np.sqrt(time_to_expiry)) - \
                risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2_val) + \
                dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1_val)
        
        # Theta par jour (en divisant par 365)
        return theta / 365.0

    def theta_put(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le Theta d'un put.
        
        Returns:
            float: Theta de l'option put (pour un jour)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        d2_val = self.d2(d1_val, volatility, time_to_expiry)
        
        theta = -spot * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1_val) * volatility / \
                (2 * np.sqrt(time_to_expiry)) + \
                risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2_val) - \
                dividend_yield * spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1_val)
        
        # Theta par jour
        return theta / 365.0

    def rho_call(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le Rho d'un call (sensibilité du prix par rapport au taux d'intérêt).
        
        Returns:
            float: Rho de l'option call (pour un changement de taux d'intérêt de 1%)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        d2_val = self.d2(d1_val, volatility, time_to_expiry)
        
        # Rho pour un changement de 1% du taux d'intérêt
        return 0.01 * strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2_val)

    def rho_put(self, spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Calcule le Rho d'un put.
        
        Returns:
            float: Rho de l'option put (pour un changement de taux d'intérêt de 1%)
        """
        if time_to_expiry <= 0:
            return 0.0
        
        d1_val = self.d1(spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield)
        d2_val = self.d2(d1_val, volatility, time_to_expiry)
        
        # Rho pour un changement de 1% du taux d'intérêt
        return -0.01 * strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2_val)


class ImpliedVolatility:
    """
    Classe pour calculer la volatilité implicite à partir des prix d'options observés.
    """
    
    def __init__(self, max_iterations=100, precision=0.00001):
        """
        Initialise la classe de calcul de volatilité implicite.
        
        Args:
            max_iterations (int): Nombre maximal d'itérations pour la méthode de Newton-Raphson
            precision (float): Précision souhaitée pour la convergence
        """
        self.max_iterations = max_iterations
        self.precision = precision
        self.bs_model = BlackScholesModel()
        
    def calculate_call(self, market_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield=0, initial_vol=0.2):
        """
        Calcule la volatilité implicite pour une option d'achat (call).
        
        Args:
            market_price (float): Prix de marché de l'option
            spot (float): Prix du sous-jacent
            strike (float): Prix d'exercice
            time_to_expiry (float): Temps jusqu'à l'expiration en années
            risk_free_rate (float): Taux sans risque
            dividend_yield (float): Rendement du dividende
            initial_vol (float): Volatilité initiale pour commencer l'itération
            
        Returns:
            float: Volatilité implicite calculée
        """
        if time_to_expiry <= 0:
            # Pour les options à expiration, la vol implicite n'a pas de sens
            return float('nan')
        
        # Valeur intrinsèque
        intrinsic = max(0, spot - strike)
        if market_price <= intrinsic:
            # Si le prix de marché est inférieur ou égal à la valeur intrinsèque,
            # pas de valeur temps donc vol implicite indéfinie
            return float('nan')
        
        vol = initial_vol
        for i in range(self.max_iterations):
            price = self.bs_model.call_price(spot, strike, time_to_expiry, risk_free_rate, vol, dividend_yield)
            vega = self.bs_model.vega(spot, strike, time_to_expiry, risk_free_rate, vol, dividend_yield) / 0.01  # Ajuster le vega
            
            price_diff = market_price - price
            
            if abs(price_diff) < self.precision:
                return vol
            
            if abs(vega) < 1e-10:  # Éviter la division par zéro
                break
                
            vol = vol + price_diff / vega
            
            if vol <= 0:  # Volatilité ne peut pas être négative
                vol = 0.001
                
        return float('nan')  # Pas de convergence
    
    def calculate_put(self, market_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield=0, initial_vol=0.2):
        """
        Calcule la volatilité implicite pour une option de vente (put).
        
        Args:
            market_price (float): Prix de marché de l'option
            spot (float): Prix du sous-jacent
            strike (float): Prix d'exercice
            time_to_expiry (float): Temps jusqu'à l'expiration en années
            risk_free_rate (float): Taux sans risque
            dividend_yield (float): Rendement du dividende
            initial_vol (float): Volatilité initiale pour commencer l'itération
            
        Returns:
            float: Volatilité implicite calculée
        """
        if time_to_expiry <= 0:
            return float('nan')
        
        # Valeur intrinsèque
        intrinsic = max(0, strike - spot)
        if market_price <= intrinsic:
            return float('nan')
        
        vol = initial_vol
        for i in range(self.max_iterations):
            price = self.bs_model.put_price(spot, strike, time_to_expiry, risk_free_rate, vol, dividend_yield)
            vega = self.bs_model.vega(spot, strike, time_to_expiry, risk_free_rate, vol, dividend_yield) / 0.01
            
            price_diff = market_price - price
            
            if abs(price_diff) < self.precision:
                return vol
            
            if abs(vega) < 1e-10:
                break
                
            vol = vol + price_diff / vega
            
            if vol <= 0:
                vol = 0.001
                
        return float('nan')
