from src.model.transformer import generate_and_decode_text, get_model

if __name__ == "__main__":
    model = get_model()
    generate_and_decode_text(model, 100)
