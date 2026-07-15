import numpy as np


class AnomalyDetector:

    def __init__(self, df):
        self.df = df

    def summary_report(self):
        """
        Génère un rapport synthétique sur la qualité du dataset.
        """

        total_rows = self.df.shape[0]
        total_cols = self.df.shape[1]

        # Variables
        categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).shape[1]

        numerical_cols = self.df.select_dtypes(
            include=["number"]
        ).shape[1]

        # Doublons
        duplicates = self.df.duplicated().sum()

        # Colonnes constantes
        constant_cols = (
            self.df.nunique(dropna=False) == 1
        ).sum()

        # Valeurs manquantes
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]

        print("=" * 65)
        print("             RAPPORT DE DÉTECTION DES ANOMALIES")
        print("=" * 65)

        print("\nInformations générales")
        print("-" * 65)
        print(f"Nombre d'observations : {total_rows}")
        print(f"Nombre de variables   : {total_cols}")

        print("\nQualité des données")
        print("-" * 65)
        print(f"Doublons               : {duplicates}")
        print(f"Colonnes constantes    : {constant_cols}")
        print(f"Variables catégorielles: {categorical_cols}")
        print(f"Variables numériques   : {numerical_cols}")

        print("\nValeurs manquantes")
        print("-" * 65)

        if missing.empty:
            print("Aucune valeur manquante détectée.")
        else:
            print(f"Colonnes concernées : {len(missing)}\n")

            for col, value in missing.items():

                percent = (value / total_rows) * 100

                print(
                    f"{col:<50} {value:>3} ({percent:.1f} %)"
                )

        print("\nConclusion")
        print("-" * 65)

        if duplicates == 0 and constant_cols == 0:
            print("Le jeu de données présente une bonne qualité globale.")
        else:
            print("Le jeu de données nécessite un nettoyage.")

        if not missing.empty:
            print("Les principales anomalies concernent les valeurs manquantes.")
        else:
            print("Aucune anomalie importante n'a été détectée.")

        print("Le dataset est prêt pour l'étape de nettoyage.")

        print("=" * 65)

    def run(self):
        self.summary_report()
