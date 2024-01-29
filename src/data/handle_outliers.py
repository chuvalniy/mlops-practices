import click
import pandas as pd


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def handle_outliers(input_path: str, output_path: str) -> None:
    """
    Replaces outliers with the closest values to feature boundary using interquartile range method.
    :param input_path: Path to read .csv file.
    :param output_path: Path to save .csv file.
    :return:
    """
    df = pd.read_csv(input_path)

    # Columns that have outliers.
    outlier_columns = ['SGOT_AST', 'SGOT_ALT', 'BLDS', 'tot_chole', 'triglyceride', 'serum_creatinine', 'waistline',
                       'SBP', 'DBP']

    for col in outlier_columns:
        # Calculate quartiles for the feature.
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1  # Interquartile range.

        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)  # Replace outliers with boundary values.

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    handle_outliers()
