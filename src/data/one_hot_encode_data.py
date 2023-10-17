import click
import pandas as pd


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def one_hot_encode_data(input_path: str, output_path: str) -> None:
    """
    Converts categorical features into number using one-hot encoding representation.
    :param input_path: Path to read .csv file.
    :param output_path: Path to save .csv file with converted data.
    :return: None
    """

    df = pd.read_csv(input_path)
    df = pd.get_dummies(df)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    one_hot_encode_data()
