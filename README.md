# Image Restoration via Deep Image Prior (U-Net) 

Ce projet implémente une architecture U-Net pour réaliser des tâches de restauration d'image (Inpainting, Débruitage) **sans aucun jeu de données d'entraînement externe**.

Il repose sur le concept de **Deep Image Prior (DIP)**.Le principe repose sur le fait que l'architecture du réseau est naturellement incapable de générer du bruit aléatoire facilement. Elle apprend donc d'abord à reconstruire les formes et les couleurs (l'image propre) avant de réussir à capturer le bruit. En arrêtant l'entraînement au bon moment, on obtient ainsi une image restaurée.

## Fonctionnalités

Le script propose trois expériences distinctes :

1.  **Reconstruction :** Le réseau apprend à reproduire une image cible à partir d'un bruit d'entrée fixe.
2.  **Inpainting (Remplissage) :** Le réseau devine et reconstruit les pixels manquants d'une image masquée.
3.  **Débruitage :** Le réseau filtre une image corrompue par un bruit gaussien en exploitant sa résistance naturelle aux hautes fréquences aléatoires.

## Architecture Technique

* **Modèle :** U-Net  (Encoder-Decoder) avec 4 niveaux de profondeur.
* **Skip Connections :** Utilisées pour réinjecter les détails spatiaux de l'encodeur vers le décodeur.
* **Entrée ($z$) :** Un tenseur de bruit aléatoire fixe. On optimise les poids du réseau pour transformer ce bruit en l'image cible.
