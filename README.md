# Image Restoration via Deep Image Prior (U-Net) 

Ce projet implémente une architecture U-Net pour réaliser des tâches de restauration d'image (Inpainting, Débruitage) **sans aucun jeu de données d'entraînement externe**.

Il repose sur le concept de Deep Image Prior (DIP) : l'architecture du réseau de convolution elle-même agit comme une régularisation ("prieur"). Le réseau apprend plus vite à générer une image naturelle cohérente que du bruit aléatoire, ce qui permet de restaurer l'image avant d'apprendre les défauts.

## Fonctionnalités

Le script propose trois expériences distinctes :

1.  **Reconstruction :** Le réseau apprend à reproduire une image cible à partir d'un bruit d'entrée fixe.
2.  **Inpainting (Remplissage) :** Le réseau devine et reconstruit les pixels manquants d'une image masquée.
3.  **Débruitage :** Le réseau filtre une image corrompue par un bruit gaussien en exploitant sa résistance naturelle aux hautes fréquences aléatoires.

## Architecture Technique

* **Modèle :** U-Net  (Encoder-Decoder) avec 4 niveaux de profondeur.
* **Skip Connections :** Utilisées pour réinjecter les détails spatiaux de l'encodeur vers le décodeur.
* **Entrée ($z$) :** Un tenseur de bruit aléatoire fixe. On optimise les poids du réseau pour transformer ce bruit en l'image cible.