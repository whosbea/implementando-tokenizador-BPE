from transformers import AutoTokenizer


def load_wordpiece_tokenizer():
    """
    Carrega o tokenizador multilíngue do BERT.
    """
    return AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def tokenize_test_sentence(tokenizer, text: str) -> list[str]:
    """
    Tokeniza a frase de teste usando WordPiece.
    """
    return tokenizer.tokenize(text)