import click
import tiktoken


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def tokenize_data(input_path: str, output_path: str):
    """Reads .txt raw text and converts it a new .txt file with integer tokens.
    :param input_path: Path to read .txt file.
    :param output_path: Path to save tokenized .txt file.
    :return: None
    """

    with open(input_path, "r", encoding='utf-8') as f:
        data = f.read()

    encoder = tiktoken.encoding_for_model("gpt2")

    tokens = encoder.encode(data)

    with open(output_path, "w", encoding='utf-8') as f:
        tokens_str = " ".join(str(token) for token in tokens)
        f.write(tokens_str)


if __name__ == "__main__":
    tokenize_data()
