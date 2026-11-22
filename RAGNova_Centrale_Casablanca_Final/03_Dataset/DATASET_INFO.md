# Jeu de données et provenance

## BoolQ — version 2

- Page Kaggle : https://www.kaggle.com/datasets/averkij/boolq-dataset
- Identifiant : `averkij/boolq-dataset`, version 2.
- Licence indiquée par Kaggle : CC BY-SA 3.0.
- Archive conservée : `raw/boolq_dataset_v2.zip` (SHA-256 : `a35ea75fcee1170a85c61020c366e7f2ceea55017c0e9fddbba7f7142c63a969`).
- Éléments étiquetés après lecture des fichiers : 12 697 questions. Le fichier de test officiel comporte 3 245 questions sans champ de réponse.
- Données étiquetées : 7 907 « oui » et 4 790 « non » ; 8 208 titres et 10 144 passages distincts dans les partitions étiquetées.
- Le corpus d'indexation dédupliqué contient 12 259 passages en incluant les passages du fichier de test officiel.

BoolQ associe une question formulée en langage naturel, un passage et une réponse binaire. Il convient à un prototype de récupération fondée sur un contexte et à l'évaluation d'un lecteur oui/non ; il ne constitue pas une évaluation de génération libre ou de factualité ouverte.

## Partition et protection contre les fuites

L'évaluation repose sur une partition interne reproductible des exemples étiquetés. Le titre Wikipédia normalisé est affecté à une partition par hachage SHA-256, avec la graine 20261114 et des seuils 70/15/15. Les questions d'un même titre ne peuvent donc pas être réparties entre train, validation et test. Les réponses ne sont pas indexées. Le fichier officiel de test sans étiquettes contribue seulement ses passages au corpus de recherche ; ses questions ne sont pas notées.

Les comptes exacts de chaque partition sont écrits par `prepare_data.py` dans `04_Experiments/data/dataset_summary.json`. Le nombre « 15 942 » parfois indiqué sur les pages descriptives additionne les exemples étiquetés et le split officiel sans réponses ; il ne désigne pas un nombre d'étiquettes exploitables.

## Attribution

BoolQ est associé à Clark et al. (2019), *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions*. ACL Anthology: https://aclanthology.org/N19-1300/. Pour toute redistribution de données ou d'œuvres dérivées, vérifier les obligations de la licence CC BY-SA 3.0 et conserver les notices d'attribution.
