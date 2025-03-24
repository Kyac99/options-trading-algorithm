# Algorithme de Trading sur les Produits Dérivés

## Description
Ce projet vise à développer un algorithme de trading performant pour les produits dérivés, en intégrant des stratégies avancées telles que le Gamma Scalping, le Dispersion Trading et le Risk Reversal. L'algorithme est conçu pour s'adapter aux conditions de marché et optimiser l'exécution des ordres.

## Fonctionnalités principales
- Stratégies de trading sur options (Gamma Scalping, Dispersion Trading, Risk Reversal)
- Modèles d'analyse du Gamma, Vega et sensibilités associées
- Algorithme adaptatif pour l'optimisation de l'exécution des ordres
- Environnement de backtesting pour évaluer la robustesse des stratégies

## Structure du projet
- `/src` : Code source principal
- `/data` : Données de marché et résultats
- `/notebooks` : Notebooks Jupyter pour l'analyse et le développement
- `/backtesting` : Outils et résultats de backtesting
- `/docs` : Documentation du projet

## Installation
```bash
# Cloner le dépôt
git clone https://github.com/Kyac99/options-trading-algorithm.git
cd options-trading-algorithm

# Installer les dépendances
pip install -r requirements.txt
```

## Phases du projet
1. Définition des exigences et choix des stratégies
2. Développement du moteur de backtesting et implémentation de la stratégie
3. Optimisation et calibration de l'algorithme d'exécution
4. Tests en conditions réelles et validation des performances
5. Documentation et mise en production