\# Rapport d’optimisation post-déploiement



\## Métriques étudiées



\- temps d’inférence du modèle ;

\- temps de réponse total de l’API ;

\- taux d’erreur ;

\- nombre de requêtes ;

\- distribution des probabilités ;

\- version du modèle.



\## Goulot d’étranglement identifié



Le principal risque de ralentissement était le chargement du modèle à chaque appel.



Cette méthode aurait provoqué :



\- des lectures disque répétées ;

\- une latence plus élevée ;

\- une consommation mémoire inutile ;

\- une moins bonne capacité à traiter plusieurs requêtes.



\## Optimisation réalisée



Le modèle est chargé une seule fois au démarrage dans :



```text

app/model\_loader.py

