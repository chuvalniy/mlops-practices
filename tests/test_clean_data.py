import pathlib

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data.clean_data import clean_data


@pytest.fixture()
def mock_csv_file(tmp_path: pathlib.Path):
    """
    Creates temporary data folder with mock.csv file to use it in further test.
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()

    test_df = pd.DataFrame(
        data=[['Y', 1.0, 2.0, 1.0, 'Female'], ['N', 3.0, 1.0, 2.0, 'Male']],
        columns=['DRK_YN', 'SMK_stat_type_cd', 'hear_left', 'hear_right', 'sex']
    )

    test_csv_path = temp_dir / 'mock.csv'
    test_df.to_csv(test_csv_path, index=False)

    return str(test_csv_path)


def test_clean_data(mock_csv_file, tmp_path: pathlib.Path):
    """
    Checks whether data in .csv file was cleaned properly.
    :param mock_csv_file: File path created in mock_csv_file().
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    output_path = tmp_path / "output.csv"

    # clean_data() uses CLI, so define CliRunner as helper to run the function.
    runner = CliRunner()
    result = runner.invoke(clean_data, [mock_csv_file, str(output_path)])

    assert result.exit_code == 0

    expected_df = pd.DataFrame(
        data=[['currently drinking', 'never smoked', 'abnormal', 'normal', 'Female'],
              ['not drinking', 'currently smoking', 'normal', 'abnormal', 'Male']],
        columns=['drinking', 'smoking', 'hear_left', 'hear_right', 'sex']
    )

    test_df = pd.read_csv(output_path)

    assert test_df.equals(expected_df)
