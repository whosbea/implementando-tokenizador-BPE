# Laboratório 6 — Construindo um Tokenizador BPE e Explorando o WordPiece

## Objetivo

Este laboratório teve dois objetivos principais:

1. construir do zero o motor básico do algoritmo **BPE (Byte Pair Encoding)**;
2. utilizar a biblioteca **transformers** do Hugging Face para observar, na prática, como o **WordPiece** segmenta palavras em subpalavras.

A proposta do laboratório foi sair da arquitetura do Transformer em si e voltar para uma etapa anterior e essencial dos modelos de linguagem: a **tokenização**. Como o próprio enunciado destaca, modelos de linguagem não leem strings diretamente; eles precisam de um vocabulário e de um mapeamento de texto para tokens numéricos. Este laboratório trabalhou justamente essa etapa.

## Estrutura de pastas

```text
implementando-tokenizador-BPE/
├── main.py
├── bpe.py
├── wordpiece_demo.py
├── requirements.txt
└── README.md
```

### Descrição dos arquivos

- `main.py`  
  Arquivo principal do projeto. Executa as três tarefas do laboratório: estatísticas do BPE, loop de fusão e teste do WordPiece.

- `bpe.py`  
  Implementa o vocabulário inicial, a contagem de pares adjacentes (`get_stats`), a fusão de pares (`merge_vocab`) e o loop principal de treinamento do BPE.

- `wordpiece_demo.py`  
  Carrega o tokenizador `bert-base-multilingual-cased` e executa a tokenização da frase de teste.

- `requirements.txt`  
  Lista a dependência necessária para esta atividade.

- `README.md`  
  Documentação do laboratório.

## Como rodar o código

### 1. Criar e ativar o ambiente virtual

No macOS/Linux com fish shell:

```bash
python3 -m venv .venv
source .venv/bin/activate.fish
```

No bash/zsh:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Executar o projeto

```bash
python main.py
```

Ou, dependendo do ambiente:

```bash
python3 main.py
```

## Dependência utilizada

```txt
transformers
```

## Fundamentos teóricos

### O que é tokenização

Modelos como o Transformer não recebem texto bruto. Antes do modelo processar uma frase, ela precisa ser convertida em unidades menores chamadas **tokens**. Esses tokens podem ser:

- palavras inteiras;
- partes de palavras;
- pontuação;
- símbolos especiais.

O problema de usar apenas palavras inteiras é que o vocabulário cresce demais e palavras raras ou novas podem ficar fora do vocabulário. Por isso, algoritmos de **subpalavras** como BPE e WordPiece se tornaram padrão.

### O que é BPE

**BPE (Byte Pair Encoding)** é um algoritmo que começa com palavras quebradas em símbolos pequenos, normalmente caracteres, e vai fundindo os pares adjacentes mais frequentes. Aos poucos, isso gera subpalavras mais úteis e frequentes.

### O que é WordPiece

**WordPiece** também é um algoritmo de subpalavras, mas usa uma lógica diferente da do BPE clássico para decidir quais segmentos manter no vocabulário. Na prática, ele também quebra palavras grandes em pedaços menores para evitar problemas com vocabulário desconhecido.

## Tarefa 1 — O motor de frequências

O laboratório pede para inicializar exatamente este dicionário:

```python
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}
```

### O que significa esse dicionário

As chaves representam palavras separadas em símbolos:

- `l o w </w>` representa a palavra `low`
- `l o w e r </w>` representa `lower`
- `n e w e s t </w>` representa `newest`
- `w i d e s t </w>` representa `widest`

Os valores representam quantas vezes cada palavra aparece no corpus.

### O que significa `</w>`

O símbolo `</w>` marca **fim de palavra**. Ele é importante porque permite ao algoritmo distinguir pedaços internos de palavra de pedaços que aparecem no final da palavra.

### Função `get_stats(vocab)`

Essa função percorre o vocabulário e conta a frequência de todos os pares adjacentes de símbolos.

Exemplo:

```text
l o w </w>
```

gera os pares:

- `(l, o)`
- `(o, w)`
- `(w, </w>)`

Como `low` aparece 5 vezes, cada um desses pares recebe frequência `5`.

### Validação principal

O enunciado exige que o par:

```python
('e', 's')
```

tenha frequência máxima `9`, pois aparece:

- 6 vezes em `newest`
- 3 vezes em `widest`

No resultado obtido no projeto, essa validação foi satisfeita:

```text
Validação do par ('e', 's'): 9
```

## Tarefa 2 — O loop de fusão

Depois de encontrar o par mais frequente, o BPE faz uma **fusão**.

### O que é fusão

Fusão significa transformar dois símbolos adjacentes frequentes em um novo símbolo único.

Exemplo:

```text
('e', 's') -> 'es'
```

Assim, uma palavra como:

```text
n e w e s t </w>
```

passa a ser:

```text
n e w es t </w>
```

### Função `merge_vocab(pair, v_in)`

Essa função recebe:

- o par mais frequente;
- o vocabulário atual.

Ela substitui todas as ocorrências desse par pela versão fundida e devolve um novo vocabulário atualizado.

### Loop principal

O enunciado pediu 5 iterações do processo. Em cada iteração, o código:

1. conta as frequências dos pares;
2. seleciona o par mais frequente;
3. funde esse par;
4. imprime o vocabulário atualizado.

### Fusões realizadas no projeto

As 5 fusões foram:

```text
('e', 's')
('es', 't')
('est', '</w>')
('l', 'o')
('lo', 'w')
```

### Resultado final observado

O vocabulário final ficou assim:

```text
low </w>: 5
low e r </w>: 2
n e w est</w>: 6
w i d est</w>: 3
```

Isso mostra duas coisas importantes:

- o algoritmo aprendeu o sufixo `est</w>`;
- o algoritmo também aprendeu a subpalavra `low`.

Essas duas formações são coerentes com a ideia central do BPE: construir pedaços frequentes e úteis de forma incremental.

## Tarefa 3 — WordPiece na prática

Nesta parte, foi utilizado o tokenizador:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
```

A frase escolhida pelo enunciado foi:

```text
Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar.
```

### Resultado obtido

```python
['Os', 'hip', '##er', '-', 'par', '##âm', '##etros', 'do', 'transform', '##er', 'são', 'in', '##cons', '##tit', '##uc', '##ional', '##mente', 'di', '##f', '##í', '##cei', '##s', 'de', 'aj', '##usta', '##r', '.']
```

## O que significa `##` nos tokens WordPiece

No WordPiece, o prefixo `##` indica que aquele token é uma **continuação da palavra anterior**, e não o começo de uma nova palavra.

Exemplos do resultado:

- `hip` + `##er` → `hiper`
- `transform` + `##er` → `transformer`
- `in` + `##cons` + `##tit` + `##uc` + `##ional` + `##mente` → `inconstitucionalmente`

Isso significa que o token sem `##` pode iniciar uma palavra, enquanto os tokens com `##` só fazem sentido como continuação de um pedaço anterior.

## Por que subpalavras evitam travamento com vocabulário desconhecido

Se o modelo dependesse apenas de palavras inteiras, uma palavra muito rara ou nova poderia não existir no vocabulário.

Com subpalavras, isso deixa de ser um problema grave.

Mesmo que a palavra inteira não exista, o tokenizador pode quebrá-la em partes conhecidas. Por exemplo, a palavra `inconstitucionalmente` foi dividida em vários pedaços menores que o modelo consegue processar. Isso impede que o sistema “trave” diante de vocabulário desconhecido e permite generalização melhor para palavras novas.

## Perguntas e respostas

### Para o BPE, como saber quando parar de juntar?

Essa é uma das decisões mais importantes do algoritmo.

Na prática, o BPE não “sabe” sozinho o ponto ideal. É preciso definir um **critério de parada**. Os critérios mais comuns são:

- parar após um número fixo de fusões;
- parar quando o vocabulário atingir um tamanho desejado;
- parar quando a frequência do melhor par ficar baixa demais e a fusão deixar de ser útil.

O risco real é exatamente isso:

- **se juntar demais**, o algoritmo começa a formar tokens muito específicos, quase palavras inteiras, e perde a vantagem das subpalavras;
- **se juntar de menos**, os tokens ficam genéricos demais, parecendo caracteres isolados, e o modelo perde informação semântica útil.

Então o ponto ideal é um equilíbrio entre:

- vocabulário compacto;
- tokens suficientemente informativos;
- capacidade de lidar com palavras raras.

No laboratório, o critério de parada foi artificial e simples: **5 iterações**, porque o objetivo era didático.

### No WordPiece, como a separação de subpalavras é feita? Como ele sabe que o correto é `##cons` e não `##const`?

O WordPiece não decide isso por “intuição linguística”. Ele decide com base no **vocabulário aprendido no treinamento do tokenizador**.

Em outras palavras:

- o tokenizador já vem com um vocabulário pronto;
- ao receber uma palavra nova, ele tenta segmentá-la usando os pedaços que existem nesse vocabulário;
- normalmente ele busca o **maior pedaço possível que esteja presente no vocabulário**, respeitando a posição na palavra.

Então, se ele gerou `##cons` em vez de `##const`, isso significa que:

- `##cons` está no vocabulário;
- `##const` provavelmente não está, ou não foi a melhor continuação válida naquele ponto da segmentação.

Ou seja, a decisão não é “qual parece mais bonita”, mas sim “qual sequência de subpalavras do vocabulário consegue cobrir a palavra de forma válida”.

## Conclusão

Este laboratório mostrou, de forma prática, como funciona a tokenização por subpalavras em dois níveis:

- no **BPE**, implementando manualmente a contagem de pares e o processo de fusão;
- no **WordPiece**, observando como um tokenizador real segmenta palavras longas em partes menores.

O resultado final deixa claro que algoritmos de subpalavras são essenciais para modelos de linguagem porque:

- controlam o tamanho do vocabulário;
- ajudam no tratamento de palavras raras;
- preservam partes úteis das palavras;
- evitam o colapso diante de vocabulário desconhecido.

## Referência

VASWANI, Ashish et al. **Attention Is All You Need**. 2017.

## Informações importantes

Este projeto contou com apoio de IA (ChatGPT 5.4 Thinking) na geração e organização do código. Todo o conteúdo foi revisado, ajustado e estudado por Beatriz Barreto.
