"""
Module de prétraitement des données.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# Nettoyage des colonnes
# ==========================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms des colonnes.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return df


# ==========================================================
# Nettoyage des données textuelles
# ==========================================================

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les colonnes textuelles.
    """

    df = df.copy()

    object_columns = df.select_dtypes(include="object").columns

    for column in object_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        df[column] = df[column].replace(
            {
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
            }
        )

    return df


# ==========================================================
# Traitement des valeurs manquantes
# ==========================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traite les valeurs manquantes selon les règles métier.
    """

    df = df.copy()

    print("=" * 70)
    print("TRAITEMENT DES VALEURS MANQUANTES")
    print("=" * 70)

    total_before = df.isna().sum().sum()

    # ------------------------------------------------------
    # Questions concernant les enfants
    # ------------------------------------------------------

    children_columns = [
        "Âge des enfants (si applicable) :",
        "Fréquence de jeu de vos enfants (par semaine) :",
        "Où vos enfants jouent-ils habituellement aux jeux vidéo ? (Sélection multiple possible)",
    ]

    if "Nombre d'enfants :" in df.columns:

        no_children = (
            pd.to_numeric(
                df["Nombre d'enfants :"],
                errors="coerce"
            )
            .fillna(0)
            .eq(0)
        )

        for column in children_columns:

            if column in df.columns:

                df.loc[
                    no_children & df[column].isna(),
                    column
                ] = "Non applicable"

    # ------------------------------------------------------
    # Questions concernant les jeux vidéo
    # ------------------------------------------------------

    if (
        "Jouez-vous aux jeux vidéo ?" in df.columns
        and
        "Fréquence de jeu (par semaine) :" in df.columns
    ):

        no_player = (
            df["Jouez-vous aux jeux vidéo ?"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["non", "no"])
        )

        df.loc[
            no_player &
            df["Fréquence de jeu (par semaine) :"].isna(),
            "Fréquence de jeu (par semaine) :"
        ] = "Non applicable"

    total_after = df.isna().sum().sum()

    print(f"Valeurs manquantes avant : {total_before}")
    print(f"Valeurs manquantes après : {total_after}")

    remaining = df.isna().sum()
    remaining = remaining[remaining > 0]

    if not remaining.empty:

        print("\nValeurs manquantes restantes :")
        print(remaining)

    else:

        print("\nAucune valeur manquante restante.")

    print("=" * 70)

    return df


# ==========================================================
# Validation
# ==========================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Vérifie la qualité du dataset après prétraitement.
    """

    print("=" * 70)
    print("VALIDATION DU DATASET")
    print("=" * 70)

    print(f"Lignes      : {df.shape[0]}")
    print(f"Colonnes    : {df.shape[1]}")
    print(f"Doublons    : {df.duplicated().sum()}")

    missing = df.isna().sum()

    if missing.sum() == 0:

        print("Valeurs manquantes : Aucune")

    else:

        print("\nValeurs manquantes :")
        print(missing[missing > 0])

    constant_columns = df.columns[df.nunique(dropna=False) == 1]

    if len(constant_columns):

        print("\nColonnes constantes :")
        print(list(constant_columns))

    print("=" * 70)


# ==========================================================
# Sauvegarde
# ==========================================================

def save_dataset(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Sauvegarde le dataset prétraité.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Dataset sauvegardé : {output_path}")


# ==========================================================
# Pipeline complet
# ==========================================================

def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exécute le pipeline complet de prétraitement.
    """

    print("\nDébut du prétraitement...\n")

    df = clean_column_names(df)

    df = clean_text_columns(df)

    df = handle_missing_values(df)

    validate_dataset(df)

    print("\nPrétraitement terminé.\n")

    return df