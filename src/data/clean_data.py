import click
import pandas as pd


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def clean_data(input_path: str, output_path: str) -> None:
    """
    Renames some of the dataframe features and their values for future clarity and converts them to the proper type.
    :param input_path: Path to read .csv file.
    :param output_path: Path to save renamed .csv file.
    :return: None
    """

    df = pd.read_csv(input_path)

    # Rename features for clarity
    df = df.rename(columns={"DRK_YN": "drinking", 'SMK_stat_type_cd': 'smoking'})

    # Transform features into proper type.
    cat_features = ['drinking', 'smoking', 'hear_left', 'hear_right', 'sex']
    df[cat_features] = df[cat_features].astype(str)  # Transform into categorical features.

    # Rename feature values for clarity.
    df['smoking'] = df['smoking'].replace({'1.0': 'never smoked', '2.0': 'quit smoking', '3.0': 'currently smoking'})
    df['hear_left'] = df['hear_left'].replace({'1.0': 'normal', '2.0': 'abnormal'})
    df['hear_right'] = df['hear_right'].replace({'1.0': 'normal', '2.0': 'abnormal'})
    df['drinking'] = df['drinking'].replace({"Y": "currently drinking", "N": "not drinking"})

    # Save into new .csv file.
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    clean_data()
