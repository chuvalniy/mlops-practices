import pathlib

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from numpy.random import RandomState

from src.data.handle_outliers import handle_outliers

random_state = RandomState(42)  # For test & expected array initial reproducibility.


@pytest.fixture()
def mock_csv_file(tmp_path: pathlib.Path):
    """
    Creates temporary data folder with mock.csv file to use it in further test.
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()

    outlier_columns = ['SGOT_AST', 'SGOT_ALT', 'BLDS', 'tot_chole', 'triglyceride', 'serum_creatinine', 'waistline',
                       'SBP', 'DBP']
    test_df = pd.DataFrame(random_state.normal(size=(100, len(outlier_columns))), columns=outlier_columns)

    mean_values = test_df.mean()
    test_df[0:3] = mean_values * 100_000

    test_csv_path = temp_dir / 'mock.csv'
    test_df.to_csv(test_csv_path, index=False)

    return str(test_csv_path)


def test_handle_outliers(mock_csv_file, tmp_path: pathlib.Path):
    """
    Tests if array after outlier removal is close to initial mean value for every feature.
    :param mock_csv_file:
    :param tmp_path:
    :return:
    """

    output_path = tmp_path / "output.csv"

    # remove_outliers() uses CLI, so define CliRunner as helper to run the function.
    runner = CliRunner()
    result = runner.invoke(handle_outliers, [mock_csv_file, str(output_path)])

    assert result.exit_code == 0

    test_df = pd.read_csv(output_path)
    expected_df = pd.DataFrame(random_state.normal(size=(100, len(test_df.columns))), columns=test_df.columns)

    # Compares every mean value for every feature from both test_df and expected_df.
    assert np.allclose(test_df.mean(), expected_df.mean(), atol=5e-1, equal_nan=True)
