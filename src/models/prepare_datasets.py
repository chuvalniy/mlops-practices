import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('train_path', type=click.Path())
@click.argument('val_path', type=click.Path())
def prepare_datasets(input_path: str, train_path: str, val_path: str) -> None:
    """
    Splits processed dataframe into train and validation sets and saves them separately.
    :param input_path: Path to read .csv file.
    :param train_path: Path to save .csv train data.
    :param val_path:  Path to save .csv validation data.
    :return: None
    """

    df = pd.read_csv(input_path)

    df_train, df_val = train_test_split(df, test_size=0.3, random_state=42)

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)


if __name__ == "__main__":
    prepare_datasets()
