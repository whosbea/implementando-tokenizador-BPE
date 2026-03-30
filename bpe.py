from collections import defaultdict


def get_initial_vocab() -> dict[str, int]:
    """
    Retorna exatamente o vocabulário pedido no laboratório.
    """
    return {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "n e w e s t </w>": 6,
        "w i d e s t </w>": 3,
    }


def get_stats(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    """
    Conta a frequência de todos os pares adjacentes de símbolos.
    """
    pairs = defaultdict(int)

    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += freq

    return dict(pairs)


def merge_vocab(pair: tuple[str, str], v_in: dict[str, int]) -> dict[str, int]:
    """
    Faz a fusão de um par adjacente no vocabulário inteiro.
    Ex.: ('e', 's') -> 'es'
    """
    v_out = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)

    for word, freq in v_in.items():
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = freq

    return v_out


def run_bpe_training(vocab: dict[str, int], num_merges: int = 5) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """
    Executa o loop principal do BPE por num_merges iterações.
    Retorna o vocabulário final e a lista de pares fundidos.
    """
    merges_done = []
    current_vocab = vocab.copy()

    for iteration in range(1, num_merges + 1):
        stats = get_stats(current_vocab)

        if not stats:
            print(f"Iteração {iteration}: nenhum par restante para fundir.")
            break

        best_pair = max(stats, key=stats.get)
        merges_done.append(best_pair)

        print(f"\nIteração {iteration}")
        print(f"Par mais frequente: {best_pair} -> frequência {stats[best_pair]}")

        current_vocab = merge_vocab(best_pair, current_vocab)

        print("Vocabulário após fusão:")
        for word, freq in current_vocab.items():
            print(f"  {word}: {freq}")

    return current_vocab, merges_done