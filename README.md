\# Projet 6 — Confirmez vos compétences en MLOps (Partie 2/2)



\## Présentation



L’objectif est de mettre à disposition une API capable de :



\- recevoir les données d’un client ;

\- calculer une probabilité de défaut ;

\- appliquer un seuil métier ;

\- retourner une décision de crédit ;

\- enregistrer les appels réalisés en production ;

\- surveiller la latence et les erreurs ;

\- détecter une éventuelle dérive des données.



\## Modèle utilisé



Le modèle final est un pipeline de machine learning enregistré dans :



```text

models/best\_lightgbm.joblib

