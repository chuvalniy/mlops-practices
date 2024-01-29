import click
import pandas as pd


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def remove_duplicates(input_path: str, output_path: str) -> None:
    """
    Removes all duplicate examples from the dataframe.
    :param input_path: Path to read .csv file.
    :param output_path: Path to save .csv file.
    :return:
    """

    df = pd.read_csv(input_path)
    df = df.drop_duplicates(keep='first')

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    remove_duplicates()
