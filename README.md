# Segmentation Clients E-Commerce — RFM Dashboard Pro

> **Dashboard décisionnel avancé** combinant l'analyse RFM classique avec le Machine Learning (K-Means) pour segmenter, comprendre et activer votre portefeuille clients e-commerce.

---

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis & Installation](#prérequis--installation)
3. [Lancement de l'application](#lancement-de-lapplication)
4. [Architecture du projet](#architecture-du-projet)
5. [Source de données](#source-de-données)
6. [Méthodologie analytique](#méthodologie-analytique)
   - [Le modèle RFM](#le-modèle-rfm)
   - [Le clustering K-Means](#le-clustering-k-means)
   - [Le nommage automatique des segments](#le-nommage-automatique-des-segments)
7. [Pages du Dashboard — Guide complet](#pages-du-dashboard--guide-complet)
   - [Page 1 — Vue Globale](#page-1--vue-globale)
   - [Page 2 — Analyse Descriptive](#page-2--analyse-descriptive)
   - [Page 3 — Segmentation RFM](#page-3--segmentation-rfm)
   - [Page 4 — Interprétation Métier](#page-4--interprétation-métier)
   - [Page 5 — Fiche Client](#page-5--fiche-client)
   - [Page 6 — Comparateur de Segments](#page-6--comparateur-de-segments)
   - [Page 7 — Simulateur What-If](#page-7--simulateur-what-if)
   - [Page 8 — Clients à Risque](#page-8--clients-à-risque)
   - [Page 9 — Export & Décisions](#page-9--export--décisions)
8. [Interprétation des Visualisations](#interprétation-des-visualisations)
9. [Segments Clients — Profils & Actions](#segments-clients--profils--actions)
10. [Prise de Décision Marketing](#prise-de-décision-marketing)
11. [Paramètres & Configuration](#paramètres--configuration)
12. [Limites & Perspectives d'amélioration](#limites--perspectives-damélioration)

---

## Vue d'ensemble

Ce dashboard a été conçu pour répondre à une question fondamentale du marketing digital :

> **"Tous mes clients ne se valent pas — comment les différencier intelligemment pour leur parler de façon pertinente ?"**

Il repose sur trois piliers :

| Pilier | Technologie | Objectif |
|---|---|---|
| **Analyse comportementale** | RFM (Recency, Frequency, Monetary) | Quantifier le comportement d'achat |
| **Machine Learning** | K-Means Clustering | Regrouper automatiquement les profils similaires |
| **Intelligence décisionnelle** | Streamlit Dashboard | Transformer les données en actions concrètes |

L'application est conçue pour être utilisée par des profils **non-techniques** (responsables marketing, CRM managers, directeurs commerciaux) autant que par des **analystes data**.

---

## Prérequis & Installation

### Environnement Python

Python **3.8 ou supérieur** est requis.

### Installation des dépendances

```bash
pip install streamlit pandas scikit-learn plotly openpyxl requests \
            geopandas pycountry-convert matplotlib seaborn reportlab
```

> **Note :** `geopandas` est optionnel. Si l'installation échoue (dépendances système complexes), la carte choroplèthe sera désactivée mais le reste du dashboard fonctionnera normalement. Une variable interne `HAS_GEO` gère cette situation.

### Vérification des installations

```bash
python -c "import streamlit, pandas, sklearn, plotly; print('OK')"
```

---

## Lancement de l'application

```bash
streamlit run dashboard_segmentation.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

### Options de lancement avancées

```bash
# Spécifier un port
streamlit run dashboard_segmentation.py --server.port 8080

# Mode headless (serveur sans navigateur)
streamlit run dashboard_segmentation.py --server.headless true

# Partager en réseau local
streamlit run dashboard_segmentation.py --server.address 0.0.0.0
```

---

### Structure interne du code

Le fichier `dashboard_segmentation.py` est organisé en blocs fonctionnels distincts :

```
[CONFIG PAGE]          → Configuration Streamlit, session state
[STYLE CSS]            → Thème dark, typographie, composants UI
[PALETTE SEGMENTS]     → Couleurs et métadonnées par segment
[CHARGEMENT DONNÉES]   → load_and_clean_data(), generate_synthetic_data()
[CALCUL RFM]           → compute_rfm()
[CLUSTERING]           → run_kmeans(), elbow_data()
[NOMMAGE SEGMENTS]     → name_segments()
[HELPERS GRAPHIQUES]   → style_fig(), kpi_card(), ...
[SIDEBAR]              → Navigation, filtres, paramètres
[PAGES]                → 9 pages indépendantes (if/elif)
[FOOTER]               → Pied de page
```

---

## Source de données

### Dataset par défaut : UCI Online Retail

Le dashboard est pré-calibré sur le **UCI Online Retail Dataset**, un jeu de données de référence en e-commerce analytics.

| Caractéristique | Valeur |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) |
| Période couverte | Déc. 2010 → Déc. 2011 |
| Transactions | ~541 000 lignes brutes |
| Pays | Principalement Royaume-Uni + exports UE |
| Format | Excel `.xlsx` |

**Colonnes attendues :**

| Colonne | Type | Description |
|---|---|---|
| `InvoiceNo` | String | Identifiant de facture (préfixe `C` = retour, exclu) |
| `StockCode` | String | Code article |
| `Description` | String | Libellé produit |
| `Quantity` | Integer | Quantité (lignes négatives = retours, exclues) |
| `InvoiceDate` | DateTime | Date et heure de la transaction |
| `UnitPrice` | Float | Prix unitaire en GBP (£) |
| `CustomerID` | String | Identifiant client (lignes sans ID exclues) |
| `Country` | String | Pays de la commande |

### Données de démonstration synthétiques

En l'absence de fichier uploadé, le dashboard génère automatiquement **25 000 transactions synthétiques** reproduisant les caractéristiques statistiques du dataset UCI (distribution des prix, répartition géographique, saisonnalité). Ces données sont reproductibles via `numpy.random.seed(42)`.

### Import de vos propres données

Via la sidebar, importez un fichier `.xlsx` ou `.csv` respectant les colonnes ci-dessus. Le code normalise automatiquement les noms de colonnes (insensible à la casse, aux espaces).

### Nettoyage automatique appliqué

Le pipeline de nettoyage effectue les opérations suivantes :

- Suppression des lignes sans `CustomerID` ou sans `InvoiceDate`
- Exclusion des factures d'annulation (préfixe `C`)
- Exclusion des quantités ≤ 0 et des prix ≤ 0
- Calcul de `TotalPrice = Quantity × UnitPrice`
- Conversion des dates au format `datetime`

---

## Méthodologie analytique

### Le modèle RFM

RFM est un modèle comportemental éprouvé, utilisé depuis les années 1990 en marketing direct. Il repose sur trois dimensions calculées **par client** :

#### R — Recency (Récence)

```
Récence = Date de référence - Date du dernier achat (en jours)
Date de référence = Date maximale dans les données + 1 jour
```

**Interprétation :** Un client avec une récence de **10 jours** est actif très récemment. Un client avec **300 jours** est potentiellement perdu. Plus la récence est **faible**, plus le client est **précieux** (il vient de se manifester).

#### F — Frequency (Fréquence)

```
Fréquence = Nombre de factures distinctes par client
```

**Interprétation :** La fréquence mesure la régularité d'achat. Un client avec une fréquence de **20** est un habitué fidèle. Un client avec une fréquence de **1** est un acheteur unique. Attention : la fréquence est calculée en nombre de commandes, non en nombre d'articles.

#### M — Monetary (Montant)

```
Montant = Somme de tous les TotalPrice par client
```

**Interprétation :** Le montant total dépensé par le client sur l'ensemble de la période. Il mesure la **valeur économique brute** du client. Attention aux outliers (quelques gros acheteurs peuvent fausser la lecture — le dashboard applique des transformations logarithmiques pour atténuer cet effet).

#### Pourquoi RFM ?

Le modèle RFM est particulièrement adapté à l'e-commerce car :
- Il est **directement calculable** depuis les données transactionnelles brutes (aucune donnée externe requise)
- Il est **interprétable** par des profils non-techniques
- Il prédit bien la **valeur future** et le **risque de churn**
- Il constitue une base solide pour l'**orchestration CRM** (email, retargeting, programmes de fidélité)

---

### Le clustering K-Means

#### Principe

K-Means est un algorithme de clustering non supervisé qui regroupe les clients en `k` groupes de façon à **minimiser la variance intra-cluster** (les clients d'un même groupe se ressemblent) et **maximiser la variance inter-clusters** (les groupes sont distincts les uns des autres).

#### Pré-traitements appliqués

Avant l'entraînement, deux transformations sont appliquées pour améliorer la qualité du clustering :

**1. Transformation logarithmique** des variables asymétriques :

```python
X["Frequency"] = np.log1p(X["Frequency"])
X["Monetary"]  = np.log1p(X["Monetary"])
```

La fréquence et le montant suivent des distributions très asymétriques à droite (quelques clients dépensent beaucoup). La transformation `log(1 + x)` ramène ces distributions vers une forme plus gaussienne, ce qui améliore significativement la qualité du clustering.

**2. Standardisation** (Z-score via `StandardScaler`) :

```python
X_scaled = StandardScaler().fit_transform(X)
```

K-Means est sensible à l'échelle des variables. Sans standardisation, la variable `Monetary` (qui vaut des centaines de £) dominerait la distance euclidienne par rapport à `Frequency` (quelques unités). La standardisation place chaque variable sur la même échelle (moyenne 0, écart-type 1).

#### Hyperparamètres

| Paramètre | Valeur | Justification |
|---|---|---|
| `n_init` | 10 | 10 initialisations aléatoires, meilleure solution retenue |
| `max_iter` | 300 | Nombre max d'itérations avant convergence forcée |
| `random_state` | 42 | Reproductibilité des résultats |

#### Choix du nombre de clusters (k)

Le paramètre `k` est configurable via le slider de la sidebar (2 à 8). Deux métriques guident le choix optimal :

- **Inertie (méthode du coude)** : Somme des distances² de chaque point à son centroïde. On cherche le point d'inflexion (le "coude") où ajouter un cluster n'apporte plus de gain significatif.
- **Score Silhouette** : Mesure la cohésion et la séparation des clusters. Varie entre -1 et 1. Un score > 0.5 indique des clusters bien définis. On cherche à le maximiser.

**Recommandation pratique :** Pour un dataset e-commerce typique, **k = 4** offre un bon compromis entre précision et interprétabilité métier. Augmenter k au-delà de 6 rend l'interprétation difficile sans gain décisionnel significatif.

---

### Le nommage automatique des segments

Une fois le clustering effectué, chaque cluster reçoit un nom métier lisible. L'algorithme de nommage calcule un **score composite** par cluster :

```python
Score = - Recency_moy / Recency_max       # Inactivité pénalisée
        + Frequency_moy / Frequency_max   # Fréquence valorisée
        + Monetary_moy / Monetary_max     # Montant valorisé
```

Les clusters sont ensuite classés par score décroissant et reçoivent les labels suivants dans l'ordre : `Champions`, `Clients Fidèles`, `Clients Prometteurs`, `À Risque`, `Clients Perdus`, `Occasionnels`.

---

## Pages du Dashboard — Guide complet

### Page 1 — Vue Globale

**Objectif :** Obtenir une photographie instantanée de la santé du portefeuille clients.

#### KPIs principaux (5 indicateurs)

| KPI | Formule | Usage |
|---|---|---|
| **Clients** | `CustomerID.nunique()` | Taille réelle du portefeuille |
| **CA Total** | `sum(Quantity × UnitPrice)` | Performance financière globale |
| **Commandes** | `InvoiceNo.nunique()` | Volume d'activité |
| **Panier moyen** | `CA Total / Nb Commandes` | Valeur unitaire de transaction |
| **CA / Client** | `CA Total / Nb Clients` | Valeur moyenne par client |

#### Heatmap d'activité calendaire

Visualisation inspirée des graphes de contribution GitHub. Chaque cellule représente un jour, colorée selon le chiffre d'affaires généré ce jour-là.

**Comment la lire :** Les zones rouge/orange intense correspondent aux pics d'activité (promotions, fêtes, soldes). Les zones grises ou absentes indiquent une inactivité (weekends, jours fériés). Ce graphe permet d'identifier immédiatement les **saisonnalités** et les **anomalies** (chutes de CA inattendues).

**Décision associée :** Planifier les campagnes CRM en anticipation des pics. Éviter les relances pendant les creux structurels (ex : Noël pour certaines catégories).

#### Évolution mensuelle du CA

Graphe en barres avec tendance lissée sur 3 mois (moyenne mobile).

**Comment le lire :** Les barres montrent le CA brut mois par mois. La ligne en pointillés orange lisse les variations pour révéler la tendance de fond. Un CA en croissance avec des barres progressivement plus hautes indique une dynamique positive. Une divergence entre la tendance et les barres signale une forte saisonnalité.

**Décision associée :** Identifier les mois creux pour intensifier les actions de réactivation. Identifier les mois de pic pour renforcer la capacité logistique et les stocks.

#### Carte choroplèthe mondiale

Carte géographique colorée par intensité de CA par pays.

**Comment la lire :** Plus la couleur d'un pays est sombre (bleu foncé), plus son contribution au CA est élevée. Cette carte révèle instantanément la concentration géographique du business.

**Décision associée :** Segmenter les actions marketing par marché. Identifier les marchés sous-exploités à fort potentiel. Adapter la communication (langue, devise, offres locales).

#### Articles populaires & Mix produits

**Comment les lire :** Le graphe en barres classe les 8 produits par CA décroissant (gradient violet → vert). Le donut chart montre leur part relative dans le CA total.

**Décision associée :** Protéger et mettre en avant les produits à fort CA. Évaluer la dépendance à quelques références (risque de concentration). Identifier des opportunités de cross-sell.

---

### Page 2 — Analyse Descriptive

**Objectif :** Comprendre la distribution des comportements avant de segmenter.

#### Histogrammes RFM (3 graphes)

Trois histogrammes indépendants montrant la distribution de chaque variable RFM.

**Comment les lire :**

- **Récence :** Si la distribution est concentrée sur les faibles valeurs (0–60 jours), votre base est active. Si elle est étalée vers la droite avec un pic sur 300+ jours, vous avez une base largement inactive.
- **Fréquence :** Typiquement très asymétrique — la majorité des clients n'achète qu'une ou deux fois. Un "long tail" vers la droite révèle l'existence de clients très fidèles (précieux à identifier et protéger).
- **Montant :** Distribution exponentielle classique. La plupart des clients dépensent peu ; quelques clients dépensent énormément. La loi de Pareto (80/20) s'applique souvent ici.

**Décision associée :** Ces distributions guident le choix de `k` (si tout le monde se ressemble, peu de segments sont utiles) et révèlent l'ampleur du travail de réactivation nécessaire.

#### CA par jour de la semaine

Graphe en barres montrant la répartition du CA selon les 7 jours de la semaine.

**Comment le lire :** Les jours les plus clairs/foncés révèlent les préférences d'achat. Pour la grande distribution, on observe souvent des pics en milieu de semaine (mardi-jeudi). Les weekends peuvent être forts pour certaines niches (décoration, loisirs) ou faibles (B2B).

**Décision associée :** Planifier les envois d'emails promotionnels et les notifications push **les jours et heures de pic d'activité** pour maximiser les taux d'ouverture et de conversion.

#### Distribution des fréquences d'achat

Graphe en barres groupant les clients selon leur nombre de commandes (1, 2, 3–5, 6–10, etc.).

**Comment le lire :** Si plus de 50 % de vos clients n'ont commandé qu'une seule fois, votre enjeu prioritaire est la **fidélisation de la première commande** (welcome sequence, programme de parrainage). Si vous avez une forte proportion de clients à 6+ commandes, vous avez une base solide à protéger.

**Décision associée :** Dimensionner le budget d'acquisition vs. le budget de rétention. Un fort taux de clients à une seule commande justifie d'investir dans les séquences d'onboarding.

#### Scatter Fréquence vs Montant (coloré par Récence)

Nuage de points croisant fréquence (axe X), montant dépensé (axe Y), et récence (couleur).

**Comment le lire :** Les points en haut à droite (haute fréquence, haut montant) sont vos meilleurs clients. Si leur couleur tire vers le vert (faible récence = actifs récemment), ils sont en excellente santé. Si leur couleur tire vers le rouge (haute récence = inactifs depuis longtemps), c'est une alerte : vos meilleurs clients partent.

**Décision associée :** Surveiller régulièrement ce graphe pour détecter un mouvement de vos meilleurs clients vers des zones de récence élevée — signal précoce de churn sur les segments à haute valeur.

---

### Page 3 — Segmentation RFM

**Objectif :** Visualiser et valider les résultats du clustering K-Means.

#### Métriques du clustering (4 KPIs)

| Métrique | Interprétation |
|---|---|
| **Segments créés** | Valeur de `k` choisie par l'utilisateur |
| **Clients segmentés** | Tous les clients avec RFM valide (100 %) |
| **Score Silhouette** | Qualité de la séparation. > 0.5 = bien séparé, > 0.7 = excellent |
| **Inertie** | Compacité interne. À minimiser — suivre son évolution en faisant varier `k` |

#### Méthode du coude (optionnelle)

Double graphe affichant l'inertie (axe gauche) et le score Silhouette (axe droit) pour k = 2 à 10.

**Comment le lire :** L'inertie décroît toujours quand k augmente. On cherche le point où la décroissance ralentit brusquement (le "coude"). Le score Silhouette, lui, atteint un maximum puis redescend. **Le k optimal est à l'intersection de ces deux signaux** : là où le coude de l'inertie coïncide avec le pic de Silhouette.

**Décision associée :** Valider le choix de k avant de communiquer les segments à l'équipe marketing. Un mauvais k produit des segments artificiels qui ne correspondent pas à de vrais profils comportementaux.

#### Donut chart des segments

Répartition des clients par segment (pourcentage et volume).

**Comment le lire :** Un segment qui dépasse 50 % du total peut indiquer un k trop faible (un grand groupe homogène n'est pas encore subdivisé). À l'inverse, un segment avec moins de 3–5 % des clients peut être statistiquement fragile et difficile à activer en campagne.

**Décision associée :** S'assurer que chaque segment est assez large pour justifier une activation dédiée (un segment de 50 clients ne justifie pas une campagne email distincte) mais assez distinct pour ne pas dupliquer les actions.

#### Bubble chart : Récence vs Fréquence

Graphe où chaque bulle représente un segment. La position (X = récence, Y = fréquence) et la taille (montant moyen) permettent de comparer visuellement les segments.

**Comment le lire :** Les **meilleures bulles** sont en bas à droite (faible récence = actifs récemment, haute fréquence = réguliers) et grandes (montant élevé). Les **bulles à risque** sont en haut à gauche (inactifs depuis longtemps, peu fréquents) et petites.

**Décision associée :** Calibrer l'intensité des actions selon la position des bulles. Un segment en bas à droite mérite des actions de fidélisation (pas de sollicitation excessive). Un segment en haut à gauche mérite des campagnes de réactivation urgentes.

#### Scatter 3D des segments

Nuage de points tridimensionnel (Récence × Fréquence × Montant) coloré par segment.

**Comment le lire :** En faisant tourner le graphe (interaction souris), on observe la séparation réelle des clusters dans l'espace à 3 dimensions. Des nuages bien séparés et compacts valident la qualité du clustering. Des nuages qui se chevauchent indiquent que les segments correspondants sont difficiles à distinguer en pratique.

**Décision associée :** Ce graphe est utile pour communiquer la pertinence du modèle à des parties prenantes qui doutent de l'approche ML. Il rend visible et tangible ce que l'algorithme a "appris".

#### Tableau statistique par segment

Tableau récapitulatif avec récence moyenne, fréquence moyenne, CA moyen, CA total et parts (% clients, % CA).

**Comment le lire :** La colonne `% CA` est souvent la plus importante. Si 10 % de vos clients génèrent 60 % du CA, ces clients méritent une attention et des ressources disproportionnées. La colonne `Récence_moy` permet d'évaluer l'urgence de réactivation par segment.

**Décision associée :** Prioriser le budget CRM en proportion du `% CA`. Un segment qui représente 5 % des clients mais 30 % du CA doit recevoir des actions de rétention premium.

---

### Page 4 — Interprétation Métier

**Objectif :** Traduire les résultats techniques en recommandations actionnables.

#### Profils clients & recommandations

Pour chaque segment identifié, une fiche synthétique présente :
- Le profil comportemental (qui est ce client ?)
- Les métriques RFM moyennes du segment
- L'action marketing recommandée

**Comment utiliser cette section :** C'est la page à partager avec l'équipe marketing et les responsables CRM. Elle fait le lien entre le modèle mathématique et les leviers d'activation concrets.

#### Matrice Valeur / Engagement

Nuage de points positionant chaque segment selon son engagement (fréquence, axe X) et sa valeur (CA moyen, axe Y). Les quadrants créés par les médianes définissent 4 zones stratégiques.

**Comment la lire :**

| Quadrant | Caractéristique | Stratégie |
|---|---|---|
| **Haut droit** | Haute valeur + Très engagés | Fidéliser, récompenser, co-créer |
| **Haut gauche** | Haute valeur + Peu engagés | Réactiver en urgence (risque de churn à fort impact) |
| **Bas droit** | Faible valeur + Très engagés | Upsell, augmenter le panier moyen |
| **Bas gauche** | Faible valeur + Peu engagés | Réactiver à faible coût ou désengager |

**Décision associée :** Allouer les ressources selon le quadrant. Le quadrant haut-gauche est le plus critique (valeur élevée en danger). Ne pas gaspiller de budget sur le quadrant bas-gauche sans test préalable.

---

### Page 5 — Fiche Client

**Objectif :** Avoir une vue micro (individuelle) pour les équipes commerciales ou le service client.

**Comment l'utiliser :**
1. Saisir un `CustomerID` dans le champ de recherche
2. Cliquer sur **Rechercher**
3. Consulter le segment d'appartenance, les métriques RFM individuelles, l'historique de factures et l'évolution mensuelle des dépenses

**Utilisation typique :**
- Avant un appel de service client : comprendre le profil du client pour personnaliser l'interaction
- Pour valider un cas spécifique : vérifier si un client "Champion" a bien été classé correctement
- Pour le sales B2B : identifier les clients à fort historique d'achat pour une approche commerciale dédiée

---

### Page 6 — Comparateur de Segments

**Objectif :** Comparer deux segments côte à côte pour affiner la stratégie différenciée.

#### Tableau comparatif

Présente les métriques clés des deux segments en regard (clients, %, récence, fréquence, CA moyen, CA total, CA/client).

**Comment le lire :** Repérer les métriques où la différence est la plus marquée — c'est sur ces axes que les actions marketing doivent être différenciées. Si deux segments ont le même CA moyen mais des récences très différentes, le seul axe de différenciation est le timing des relances.

#### Diagramme radar

Radar à 3 axes (Récence normalisée, Fréquence normalisée, Montant normalisé) superposant les deux segments.

**Comment le lire :** La surface de chaque polygone représente la "valeur globale" du segment. Un polygone plein et équilibré signifie un segment fort sur tous les axes. Un polygone avec un axe atrophié révèle la faiblesse spécifique à travailler (ex : bonne récence et fréquence mais faible montant → opportunité d'upsell).

**Décision associée :** Construire des briefs créatifs différents pour chaque segment en se basant sur les axes où ils divergent le plus.

---

### Page 7 — Simulateur What-If

**Objectif :** Tester des scénarios hypothétiques pour comprendre les frontières entre segments.

**Comment l'utiliser :**
1. Ajuster les trois curseurs (Récence, Fréquence, Montant)
2. Observer immédiatement le segment prédit et la recommandation associée
3. Visualiser la position du client simulé dans l'espace 3D par rapport aux clusters réels

**Cas d'usage pratiques :**

- **Former l'équipe marketing :** "Si un client n'achète pas depuis 90 jours avec 5 commandes et £300 dépensés, dans quel segment tombe-t-il ?"
- **Définir des règles de déclenchement CRM :** "À partir de quelle récence un client passe-t-il de 'Fidèle' à 'À Risque' ?" → Déclencher automatiquement une campagne win-back à cette frontière.
- **Valider le modèle :** Comparer les prédictions du simulateur avec la connaissance terrain des clients.

---

### Page 8 — Clients à Risque

**Objectif :** Identifier proactivement les clients susceptibles de ne plus acheter (churn).

#### Calcul du score de churn

```
Score Churn = (Recency / Recency_max) × 0.50
            - (Frequency / Frequency_max) × 0.25
            - (Monetary / Monetary_max) × 0.25

Score normalisé → [0, 100]
```

**Interprétation des poids :**
- L'inactivité récente (Récence) est le signal le plus fort de churn → pondération 50 %
- La fréquence historique et la valeur économique atténuent le risque → pondérations 25 % chacune

| Catégorie | Score | Interprétation |
|---|---|---|
| **Faible risque** | 0 – 30 | Client stable, pas d'urgence |
| **Risque moyen** | 30 – 60 | Signal d'alerte, action préventive conseillée |
| **CRITIQUE** | 60 – 100 | Churn probable, action immédiate nécessaire |

**Comment utiliser ce tableau :**
1. Filtrer sur les clients **CRITIQUE** en priorité
2. Vérifier leur CA historique (`Dépensé`) — prioriser les plus gros en premier
3. Déclencher une campagne win-back personnalisée (offre de réactivation, sondage, appel commercial si valeur élevée)

#### CA total à risque

Le KPI "CA total à risque" représente l'enjeu financier du churn : combien de CA annuel serait perdu si tous ces clients partaient. C'est l'argument budgétaire pour justifier un investissement en actions de réactivation.

**Règle de décision :** Si le CA à risque dépasse un seuil défini (ex : 10 % du CA total), escalader l'alerte vers la direction et mobiliser des ressources dédiées.

---

### Page 9 — Export & Décisions

**Objectif :** Passer de l'analyse à l'action opérationnelle.

#### Export emails par segment

Génère un fichier CSV prêt à importer dans **Mailchimp, MailerLite, Klaviyo** ou tout autre outil CRM. Chaque ligne contient `CustomerID`, `Email` (simulé dans la démo) et `Segment`.

#### Export segmentation complète

CSV complet avec les 5 colonnes : `CustomerID`, `Recency`, `Frequency`, `Monetary`, `Segment`. À importer dans votre CRM, data warehouse ou outil de BI pour des analyses complémentaires.

#### Log des décisions marketing

Outil de traçabilité pour documenter les actions décidées par segment (campagne lancée, date, type d'offre, résultats attendus). Les décisions sont stockées en session et exportables en CSV.

**Utilisation recommandée :** Compléter ce log après chaque réunion de stratégie marketing. Il sert de mémoire collective et facilite le **suivi des résultats** lors de la prochaine analyse.

---

## Interprétation des Visualisations

### Guide de lecture rapide

| Visualisation | Question à laquelle elle répond | Signal d'alerte |
|---|---|---|
| Heatmap calendaire | Quand mes clients achètent-ils ? | Zones grises croissantes = désengagement |
| CA mensuel | Ma croissance est-elle saine ? | Tendance descendante + barres irrégulières |
| Carte choroplèthe | Où sont mes clients ? | Hyper-concentration sur 1–2 pays |
| Histogramme Récence | Quelle part de ma base est active ? | Pic dominant > 180 jours |
| Histogramme Fréquence | Ai-je des clients fidèles ? | 80 %+ à fréquence = 1 |
| Donut segments | Mes segments sont-ils équilibrés ? | 1 segment > 60 % |
| Bubble chart | Quels segments prioritiser ? | Grandes bulles à droite et haut |
| Scatter 3D | Mon clustering est-il valide ? | Nuages superposés |
| Matrice valeur/engagement | Comment allouer mon budget ? | Quadrant haut-gauche peuplé |
| Radar comparatif | En quoi deux segments diffèrent-ils ? | Axes très déséquilibrés |
| Score churn | Qui vais-je perdre ? | CA à risque > 10 % du total |

---

## Segments Clients — Profils & Actions

| Segment | Profil RFM | Couleur | Action prioritaire |
|---|---|---|---|
| **Champions** | Faible récence, haute fréquence, haut montant | 🟢 Vert menthe | Programme VIP, accès anticipé, ambassadeurs |
| **Clients Fidèles** | Récence modérée, fréquence régulière | 🟣 Violet | Upsell, programme de points, offres exclusives |
| **Clients Prometteurs** | Récence faible, fréquence naissante | 🔵 Bleu ciel | Onboarding, cross-sell, remise de bienvenue |
| **À Risque** | Récence élevée, anciennement actifs | 🟠 Orange | Win-back, sondage satisfaction, offre choc |
| **Clients Perdus** | Très haute récence, faible valeur | 🔴 Rouge | Dernier email de réactivation, puis suppressions |
| **Occasionnels** | Faible récence et fréquence | 🟡 Jaune or | Promotions saisonnières, retargeting |

---

## Prise de Décision Marketing

### Workflow recommandé (mensuel)

```
1. ANALYSER      → Page "Vue Globale" : état de santé du mois
2. DIAGNOSTIQUER → Page "Analyse Descriptive" : tendances comportementales
3. SEGMENTER     → Page "Segmentation RFM" : validation du clustering
4. PRIORISER     → Page "Interprétation Métier" : matrice valeur/engagement
5. ALERTER       → Page "Clients à Risque" : churn imminent
6. ACTIVER       → Page "Export & Décisions" : campagnes + documentation
```

### Budget CRM — Règle d'allocation par segment

| Segment | Part du CA recommandée | Levier principal | Canal |
|---|---|---|---|
| Champions | > 35 % du budget | Fidélisation | Email personnalisé, SMS, appel |
| Clients Fidèles | 25 % | Rétention & upsell | Email automation, push |
| À Risque | 20 % | Réactivation urgente | Email, retargeting payant |
| Clients Prometteurs | 15 % | Nurturing | Séquence onboarding |
| Occasionnels / Perdus | 5 % | Test & abandon | Email unique, suppression liste |

---

## Paramètres & Configuration

### Sidebar — Options disponibles

| Option | Valeur par défaut | Effet |
|---|---|---|
| **Source de données** | Données synthétiques | Changer pour importer un vrai dataset |
| **Nombre de segments (k)** | 4 | Ajuster selon les résultats Silhouette/Coude |
| **Afficher méthode du coude** | Non | Activer pour valider le choix de k |
| **Filtre pays** | United Kingdom | Filtrer pour une analyse géo-spécifique |
| **Filtre période** | Toute la période | Restreindre pour une analyse temporelle |

### Conseils de configuration

- **Pour une première analyse :** Utiliser les données de démonstration avec k = 4
- **Pour la production :** Importer le vrai dataset, ajuster k selon la méthode du coude, filtrer par pays si le business est multi-marché
- **Pour les réunions :** Désactiver la méthode du coude pour simplifier l'affichage

---

## Limites & Perspectives d'amélioration

### Limites actuelles

| Limite | Explication |
|---|---|
| **Pas de données démographiques** | Le modèle repose uniquement sur le comportement transactionnel |
| **Snapshot statique** | La segmentation est calculée sur la période filtrée, sans tracking dynamique |
| **K-Means = clusters sphériques** | L'algorithme suppose des clusters de forme approximativement sphérique |
| **Emails simulés** | Dans la démo, les emails sont générés (`customer+ID@example.com`) — à remplacer par les vraies adresses |
| **Pas de pipeline automatisé** | Le dashboard est une outil d'analyse, pas un orchestrateur CRM en temps réel |

### Perspectives d'évolution

- **Intégration API CRM** : Connecter directement Mailchimp, Klaviyo ou HubSpot pour déclencher les campagnes depuis le dashboard
- **Modèle prédictif de valeur future (CLV)** : Prédire le chiffre d'affaires futur par client avec des modèles BG/NBD ou Pareto/NBD
- **Alertes automatiques** : Déclencher une notification Slack/email quand le CA à risque dépasse un seuil défini
- **Segmentation dynamique** : Recalculer automatiquement les segments chaque semaine et tracker les migrations entre segments
- **Tests A/B intégrés** : Comparer les performances de différentes stratégies par segment sur le long terme

---

## Licence & Crédits

- **Dataset :** UCI Online Retail Dataset (Dr. Daqing Chen, London South Bank University)
- **Framework :** [Streamlit](https://streamlit.io/)
- **Visualisations :** [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **ML :** [scikit-learn](https://scikit-learn.org/)

---

*Dashboard conçu pour la prise de décision marketing basée sur les données — RFM + K-Means + Intelligence Décisionnelle*
