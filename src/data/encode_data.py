import click
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def encode_data(input_path: str, output_path: str) -> None:
    """
    Converts categorical features into number using one-hot encoding representation.
    :param input_path: Path to read .csv file.
    :param output_path: Path to save .csv file with converted data.
    :return: None
    """

    df = pd.read_csv(input_path)

    # One-hot encode categorical features except the target feature.
    features_to_encode = ['smoking', 'hear_left', 'hear_right', 'sex']
    df_encoded = pd.get_dummies(df[features_to_encode])

    df = df.drop(features_to_encode, axis=1).merge(df_encoded, left_index=True, right_index=True)

    # Encode target feature to be discrete number instead of a string.
    encoder = LabelEncoder()
    encoder.fit(df['drinking'])

    df['drinking'] = encoder.transform(df['drinking'])

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    encode_data()
