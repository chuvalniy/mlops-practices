import click


@click.command()
@click.argument('train_path', type=click.Path(exists=True))
@click.argument('val_path', type=click.Path(exists=True))
@click.argument('score_path', type=click.Path())
@click.argument('artifact_path', type=click.Path())
def train_model(train_path: str, val_path: str, score_path: str, artifact_path: str) -> None:
    """
    Trains model and stores training information in MLflow.
    :param train_path: Path to get training data in .csv file.
    :param val_path: Path to get validation data in .csv file.
    :param score_path: Path to save model scores in .json file.
    :param artifact_path: Path to save model artifact in .clf file.
    :return:
    """
    pass


if __name__ == "__main__":
    train_model()
