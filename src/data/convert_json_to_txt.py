import json
import click


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def convert_json_to_txt(input_path: str, output_path: str):
    """Loads raw JSON data and converts it into .txt file.
    :param input_path: Path to read JSON file.
    :param output_path: Path to save converted to .txt file.
    :return: None
    """

    with open(input_path, 'r', encoding='utf-8') as f:
        json_raw = json.load(f)

    text = ''.join(
        episode_script for episodes in json_raw.values() for episode_script in episodes.values()
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == "__main__":
    convert_json_to_txt()
