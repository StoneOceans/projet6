# Projet 6 — Confirmez vos compétences en MLOps (Partie 2/2)



## Présentation



L’objectif est de mettre à disposition une API capable de :



- recevoir les données d’un client ;

- calculer une probabilité de défaut ;

- appliquer un seuil métier ;

- retourner une décision de crédit ;

- enregistrer les appels réalisés en production ;

- surveiller la latence et les erreurs ;

- détecter une éventuelle dérive des données.



## Modèle utilisé



Le modèle final est un pipeline de machine learning enregistré dans :



```text

models/best_lightgbm.joblib

## Limite du déploiement

Le pipeline CI/CD exécute automatiquement les tests et construit l’image Docker.

L’étape de déploiement est simulée, car aucun environnement cloud permanent n’a été utilisé dans le cadre de ce prototype. La construction réussie de l’image Docker sur GitHub Actions démontre toutefois que l’application est déployable sur une plateforme compatible avec Docker.

