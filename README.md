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

## Utilisation

### Exécution d'un backtesting
Pour exécuter un backtesting d'une stratégie sur un symbole donné :

```bash
# Exécuter un backtesting de la stratégie Gamma Scalping sur SPX
python src/main.py --symbol SPX --strategy gamma_scalping --backtest

# Exécuter un backtesting de la stratégie Dispersion Trading sur NDX
python src/main.py --symbol NDX --strategy dispersion --backtest

# Exécuter un backtesting de la stratégie Risk Reversal sur RUT
python src/main.py --symbol RUT --strategy risk_reversal --backtest
```

Options disponibles :
- `--symbol` : Symbole du sous-jacent (ex: SPX, NDX, RUT)
- `--strategy` : Stratégie à utiliser (gamma_scalping, dispersion, risk_reversal)
- `--backtest` : Mode backtest (simulation historique)
- `--log-level` : Niveau de log (DEBUG, INFO, WARNING, ERROR)
- `--log-file` : Chemin vers le fichier de log
- `--config` : Chemin vers un fichier de configuration alternatif

### Exécution en mode trading réel
Pour exécuter l'algorithme en mode trading réel :

```bash
# Exécuter la stratégie Gamma Scalping sur SPX en mode réel
python src/main.py --symbol SPX --strategy gamma_scalping

# Avec logging détaillé
python src/main.py --symbol SPX --strategy gamma_scalping --log-level DEBUG --log-file logs/trading.log
```

### Analyse avec les notebooks
Pour analyser les résultats des backtests et comparer les stratégies, vous pouvez utiliser le script d'analyse :

```bash
# Lancer un notebook Jupyter
jupyter notebook notebooks/

# Ou exécuter directement le script d'analyse
python notebooks/strategy_analysis.py
```

### Personnalisation de la configuration
Vous pouvez modifier les paramètres des stratégies et du backtesting dans le fichier `config.py` :

```python
# Exemple de modification des paramètres de la stratégie Gamma Scalping
CONFIG['strategies']['gamma_scalping']['gamma_threshold'] = 0.03
CONFIG['strategies']['gamma_scalping']['position_size'] = 150

# Exemple de modification des paramètres de backtesting
CONFIG['backtest']['start_date'] = '2023-04-01'
CONFIG['backtest']['end_date'] = '2023-06-30'
```

### Visualisation des résultats
Les résultats de backtesting sont sauvegardés dans le dossier `results/` sous forme de fichiers JSON et de graphiques :

```bash
# Liste des résultats de backtesting
ls results/

# Visualiser les graphiques (en utilisant votre visionneuse d'images)
open results/equity_SPX_gamma_scalping_20250324_205541.png
```

## Phases du projet
1. Définition des exigences et choix des stratégies
2. Développement du moteur de backtesting et implémentation de la stratégie
3. Optimisation et calibration de l'algorithme d'exécution
4. Tests en conditions réelles et validation des performances
5. Documentation et mise en production