# bot_fs

Parcours de construction d’un modèle de langage. Le Transformer dense et le MoE
utilisent les mêmes partitions de TinyStories.

## Séance 1 · Préparer les données

Commandes PowerShell avec Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_tinystories.py
```

Le script lit les récits à partir d’une révision fixe de TinyStories. Il réserve
la validation officielle au test final et retire du train officiel les récits
déjà dans le test ainsi que les doublons exacts. Il trie les empreintes
SHA-256 des récits éligibles et attribue les 5 % les plus basses à notre
validation, en arrondissant à l’entier inférieur. Ce tri décide de la
partition. À l’intérieur de chaque fichier, les récits suivent l’ordre source.

Les fichiers `train.jsonl`, `validation.jsonl` et `test.jsonl` dans
`data/tinystories/` contiennent un récit et son empreinte par ligne. Le
`manifest.json` donne la révision, la licence, les règles, les effectifs,
les tailles et les empreintes des fichiers. Avant de publier ces résultats
localement, le script relit les fichiers et vérifie les trois intersections
d’empreintes. La vérification porte sur les textes UTF-8 identiques et laisse
passer les quasi-doublons.

Les lignes vides ou composées uniquement d’espaces sont écartées. Le texte des
récits conservés reste intact. Les répétitions éventuelles au sein du test
officiel restent dans le test et sont dénombrées dans le manifeste.
Pour refaire le découpage, utiliser
`--output data/tinystories-reproduction` et comparer les manifestes.
