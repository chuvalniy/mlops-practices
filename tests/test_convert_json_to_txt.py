import json
import pathlib

import pytest
from click.testing import CliRunner

from src.data.convert_json_to_txt import convert_json_to_txt


@pytest.fixture()
def mock_json_path(tmp_path: pathlib.Path):
    """
    Creates temporary data folder with mock.json file to use it in further test.
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()

    json_data = {
        "movie_1": {"episode_1": "script_1", "episode_2": "script_2"},
        "movie_2": {"episode_3": "script_3", "episode_4": "script_4"}
    }

    json_file_path = temp_dir / "mock.json"
    with open(json_file_path, "w", encoding='utf-8') as f:
        json.dump(json_data, f)

    return str(json_file_path)


def test_convert_json_to_txt(mock_json_path: str, tmp_path: pathlib.Path):
    """
    Checks whether JSON file was converted to .txt format.
    :param mock_json_path: File path created in mock_json_path().
    :param tmp_path: Temporary file path provided via pytest API.
    :return:
    """
    output_path = tmp_path / "output.txt"

    # convert_json_to_txt() uses CLI, so define CliRunner as helper to run the function.
    runner = CliRunner()
    result = runner.invoke(convert_json_to_txt, [mock_json_path, str(output_path)])

    assert result.exit_code == 0

    with open(output_path, 'r', encoding='utf-8') as f:
        output_text = f.read()

    expected_output = "script_1script_2script_3script_4"

    assert output_text == expected_output


