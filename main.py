from bpe import get_initial_vocab, get_stats, run_bpe_training
from wordpiece_demo import load_wordpiece_tokenizer, tokenize_test_sentence


def main():
    print("=== LABORATÓRIO 6: BPE E WORDPIECE ===")

    print("\n=== TAREFA 1: MOTOR DE FREQUÊNCIAS ===")
    vocab = get_initial_vocab()

    print("Vocabulário inicial:")
    for word, freq in vocab.items():
        print(f"  {word}: {freq}")

    stats = get_stats(vocab)

    print("\nFrequências dos pares:")
    for pair, freq in sorted(stats.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {pair}: {freq}")

    print(f"\nValidação do par ('e', 's'): {stats.get(('e', 's'), 0)}")

    print("\n=== TAREFA 2: LOOP DE FUSÃO ===")
    final_vocab, merges_done = run_bpe_training(vocab, num_merges=5)

    print("\nFusões realizadas:")
    for merge in merges_done:
        print(f"  {merge}")

    print("\nVocabulário final:")
    for word, freq in final_vocab.items():
        print(f"  {word}: {freq}")

    print("\n=== TAREFA 3: WORDPIECE ===")
    tokenizer = load_wordpiece_tokenizer()

    test_sentence = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
    tokens = tokenize_test_sentence(tokenizer, test_sentence)

    print("\nFrase de teste:")
    print(test_sentence)

    print("\nTokens WordPiece:")
    print(tokens)


if __name__ == "__main__":
    main()