import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables explicatives du questionnaire.
    """

    df = df.copy()

    # ==========================================================
    # 1. Variables binaires
    # ==========================================================

    binary_mapping = {
        "Oui": 1,
        "Non": 0
    }

    binary_columns = [
        "Jouez vous aux jeux vidéo ?",
        "Avez-vous remarqué des changements de comportement chez vous ou vos enfants liés aux jeux vidéo ?",
        "Avez-vous des règles spécifiques concernant l'utilisation des jeux vidéo à la maison ?"
    ]

    for column in binary_columns:

        if column in df.columns:
            df[column] = df[column].map(binary_mapping)

    # ==========================================================
    # 2. Variables à trois modalités
    # ==========================================================

    three_class_mapping = {

        "Vos enfants jouent ils aux jeux vidéo ?": {
            "Oui": 1,
            "Non": 0,
            "Pas d'enfants": 2
        },

        "Pensez-vous que les jeux vidéo ont un impact positif ou négatif sur la communication familiale ?": {
            "Négatif": 0,
            "Neutre": 1,
            "Positif": 2
        },

        "Pensez-vous que les jeux vidéo peuvent aider à gérer le stress ou l'anxiété ?": {
            "Non": 0,
            "Oui": 1,
            "Ne sait pas": 2
        },

        "Avez-vous des préoccupations concernant l'impact des jeux vidéo sur la santé mentale de vos enfants ?": {
            "Non": 0,
            "Oui": 1,
            "Pas d'enfants": 2
        }

    }

    for column, mapping in three_class_mapping.items():

        if column in df.columns:
            df[column] = df[column].map(mapping)

    # ==========================================================
    # 3. Variables ordinales
    # ==========================================================

    ordinal_mapping = {

        "Âge :": {
            "18-24": 0,
            "25-34": 1,
            "35-44": 2,
            "45-54": 3,
            "55 et plus": 4
        },

        "Nombre d'enfants :": {
            "Aucun": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4 et plus": 4
        },

        "Niveau d'études :": {
            "Pas de diplôme": 0,
            "Diplôme de l'enseignement secondaire": 1,
            "Diplôme de l'enseignement supérieur (Bac +2/+3)": 2,
            "Licence (Bac +3/+4)": 3,
            "Master (Bac +5)": 4,
            "Doctorat (Bac +8)": 5
        },

        "Fréquence de jeu (par semaine) :": {
            "Aucun": 0,
            "Moins de 1 heure": 1,
            "1-3 heures": 2,
            "4-7 heures": 3,
            "Plus de 7 heures": 4
        },

        "Fréquence de jeu de vos enfants (par semaine) :": {
            "Non applicable": 0,
            "jamais jouer": 1,
            "Moins de 1 heure": 2,
            "1-3 heures": 3,
            "4-7 heures": 4,
            "Ne sait pas": 5
        }

    }

    for column, mapping in ordinal_mapping.items():

        if column in df.columns:
            df[column] = df[column].map(mapping)

    # ==========================================================
    # 4. Variables nominales
    # ==========================================================

    nominal_columns = [
        "Genre :",
        "Situation familiale :"
    ]

    df = pd.get_dummies(
        df,
        columns=nominal_columns,
        dtype=int
    )

    # ==========================================================
    # 5. Variables multivaluées
    # ==========================================================

    multi_columns = [
        "Où jouez vous habituellement aux jeux vidéo ? (Sélection multiple possible)",
        "Où vos enfants jouent-ils habituellement aux jeux vidéo ? (Sélection multiple possible)"
    ]

    for column in multi_columns:

        if column not in df.columns:
            continue

        values = (
            df[column]
            .fillna("")
            .str.split(", ")
        )

        mlb = MultiLabelBinarizer()

        encoded = pd.DataFrame(
            mlb.fit_transform(values),
            columns=[f"{column}_{c}" for c in mlb.classes_],
            index=df.index
        )

        df = pd.concat(
            [
                df.drop(columns=[column]),
                encoded
            ],
            axis=1
        )
        # ==========================================================
        # 6. Gestion des valeurs manquantes
        # ==========================================================
        mode=0
        for column in df.columns:

            if df[column].isna().sum() > 0:

                mode = df[column].mode()

                if not mode.empty:
                    df[column] = df[column].fillna(mode.iloc[0])

    return df
def validate_encoded_dataset(df: pd.DataFrame) -> None:
    """
    Vérifie que le jeu de données est correctement encodé.
    """

    print("=" * 60)
    print("VALIDATION DU JEU DE DONNÉES ENCODÉ")
    print("=" * 60)
    

    # Dimensions
    print(f"\nDimensions : {df.shape}")

    # Types des données
    print("\nTypes des variables :")
    print(df.dtypes.value_counts())

    # Colonnes non numériques
    object_columns = df.select_dtypes(include=["object", "string"]).columns

    print("\nColonnes non encodées :")

    if len(object_columns) == 0:
        print("✔ Toutes les variables sont numériques.")
    else:
        print("✘ Variables restantes :")
        for col in object_columns:
            print(f" - {col}")

    # Valeurs manquantes
    missing = df.isna().sum()

    print("\nValeurs manquantes :")

    if missing.sum() == 0:
        print("✔ Aucune valeur manquante.")
    else:
        print(missing[missing > 0])

    # Vérification des doublons
    duplicates = df.duplicated().sum()

    print(f"\nDoublons : {duplicates}")

    # Résumé final
    if (
        len(object_columns) == 0
        and missing.sum() == 0
        and duplicates == 0
    ):
        print("\n✅ Le jeu de données est prêt pour l'entraînement des modèles.")
    else:
        print("\n⚠️ Le jeu de données nécessite encore des corrections.")