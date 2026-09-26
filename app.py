# -*- coding: utf-8 -*-
"""
================================================================================
Pipeline de PLN baseado em regras + NLTK — Avaliações / Reclamações / Mensagens
================================================================================

Aplicação Flask full stack que resolve, em um único pipeline, as atividades:

    Atividade 2  -> frequência de palavras (contar_frequencia)
    Atividade 3  -> detecção de palavras negativas (detectar_palavras_negativas)
    Atividade 4  -> remoção de stopwords em português (remover_stopwords)
    Atividade 5  -> classificação de sentimento (classificar_sentimento)
    Atividade 6  -> palavras-chave de intenção p/ roteamento (classificar_categoria)
    Atividade 7  -> palavras mais frequentes em reclamações (contar_frequencia + tipo)
    Atividade 8  -> classificação de mensagens por setor (classificar_categoria)
    Atividade 9  -> limpeza do texto (normalizar_texto)
    Atividade 10 -> tokenização + condicional p/ análise básica de sentimento
                     (tokenizar + classificar_sentimento)

Fluxo do pipeline (ordem exigida pelo enunciado):

    texto original (PT-BR)
        -> 1. normalizar  (minúsculas + sem pontuação)            [Atividade 9]
        -> 2. traduzir PT -> EN (gratuito, sem chave de API)      [ponte p/ NLTK EN]
        -> 3. tokenizar                                           [Atividade 10]
        -> 4. remover stopwords                                   [Atividade 4]
        -> 5. contar frequência  (top N)                          [Atividades 2 e 7]
        -> 6. detectar palavras negativas                         [Atividade 3]
        -> 7. classificar sentimento (regra PT + VADER no EN)     [Atividades 5 e 10]
        -> 8. classificar categoria/setor (regras condicionais)   [Atividades 6 e 8]
        -> 9. persistir no SQLite                                 [histórico]
        -> 10. exibir resultado + histórico no frontend

POR QUE A TRADUÇÃO GRATUITA?
----------------------------
O NLTK tem suporte nativo MUITO mais forte para o inglês: o
SentimentIntensityAnalyzer do VADER só opera bem em inglês, a lematização
WordNetLemmatizer só tem dicionário em inglês e vários corpora são somente
em inglês. Como o nosso texto chega em português, usamos tradução automática
PT -> EN APENAS como ponte interna para esses recursos. As regras de negócio,
a tokenização, as stopwords e as palavras-chave continuam rodando em PT-BR.

TRADUÇÃO SEM CHAVE E SEM CUSTO — E O PROBLEMA DO HTTP 429:
O Google NÃO usa o Google Cloud Translate API aqui (serviço pago, que exige
API key); usamos apenas os endpoints públicos e gratuitos de tradução. O
detalhe que fazia a tradução falhar é o BLOQUEIO ANTI-BOT: sem um
User-Agent de navegador, esses endpoints respondem HTTP 429 com uma página
HTML "Sorry..." — e o pipeline recebia HTML no lugar do JSON,tradução vazia.
Por isso `traduzir_pt_en` percorre uma cadeia de provedores, TODOS com os
mesmos headers de navegador:

    1. clients5.google.com/translate_a/t  (endpoint do dicionário do Chrome)
    2. translate.googleapis.com/..._a/single  (endpoint JSON clássico)
    3. api.mymemory.translated.net  (provedor independente do Google, sem chave)
    4. deep-translator (GoogleTranslator) — mantido como último recurso

Todos são gratuitos e não exigem chave. Se TODOS falharem (ex.: sem
internet), o pipeline NÃO quebra: segue com a análise 100% em PT-BR e
registra "tradução indisponível".
================================================================================
"""

import json
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, render_template, request

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

# Caminho do banco SQLite. Pode ser sobrescrito por variável de ambiente
# (usado nos testes para não sujar o banco da aplicação).
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db"),
)

TOP_N = 10                 # quantas palavras entram no "top palavras"
MIN_TAMANHO_TEXTO = 3      # tamanho mínimo do texto normalizado p/ analisar
TIPO_TRADUCAO_INDISPONIVEL = "tradução indisponível"

# Tipos de texto aceitos (avaliaoção / reclamação / mensagem de atendimento).
TIPOS_VALIDOS = {"avaliacao", "reclamacao", "mensagem"}

# ==============================================================================
# LISTAS CONFIGURÁVEIS (constantes nomeadas — fáceis de editar/expandir)
# ==============================================================================

# Palavras negativas/críticas usadas na Atividade 3 (priorização de suporte)
# e no sentimento por regra (Atividade 5/10). A ordem aqui define a ordem de
# exibição em detectar_palavras_negativas (determinística).
# O casamento é por RADICAL (ver seção "COMPARAÇÃO DE PALAVRAS"), então basta
# listar a raiz uma vez: "péssimos"/"pessima" casam com "péssimo", e "pessimo"
# sem acento também.
PALAVRAS_NEGATIVAS = [
    "ruim", "péssimo", "erro", "problema", "demora", "demorou",
    "falha", "defeito", "horrível", "atraso", "atrasou", "lento",
    "travou", "trava", "quebrou", "insatisfeito", "reclamação", "cara",
    # "reclamações" entra à parte: o Snowball gera radicais DIFERENTES para o
    # singular ("reclamacão" -> "reclamaca") e o plural ("reclamações" ->
    # "reclamaco"), e a comparação por prefixo não alcança a divergência na
    # última letra. Listar as duas formas cobre o plural.
    "reclamações",
    # Raízes ampliadas para cobrir reclamações do dia a dia que, antes,
    # passavam batidas e caíam em "mensagem normal".
    "pior", "odiei", "odeio", "detesto", "irritante", "irritou",
    "lentidão", "carregando", "congelou", "travando",
    "impossível", "recusado", "recusou", "duplicado",
    "errado", "errada", "ninguém", "inutilizável", "lixo", "furada",
    "vergonha", "decepcionante", "cancelado", "perdi",
]

# Expressões negativas compostas (com espaço). Não podem ser detectadas
# token a token — "não" vira stopword na etapa 4 e "funciona" sozinho não
# significa nada — então são procuradas no texto já normalizado, como as
# frases do roteador. "não funciona" é, na prática, a reclamação mais comum
# em suporte; sem esta lista ela caía em "mensagem normal".
PALAVRAS_NEGATIVAS_FRASES = [
    "não funciona", "não abre", "não consigo", "não carrega", "não conecta",
    "não responde", "não dá", "não pagou", "não recebi", "não autoriza",
    "sem resposta", "sem acesso", "travando direto",
]

# Palavras positivas usadas no sentimento por regra (Atividade 5/10).
PALAVRAS_POSITIVAS = [
    "bom", "ótimo", "excelente", "perfeito", "recomendo", "amei",
    "gostei", "adorei", "maravilhoso", "rápido", "prático", "top",
    "parabéns", "nota",
]

# Palavras-chave de roteamento por setor (Atividades 6 e 8).
# Palavras simples são procuradas nos tokens limpos; frases (com espaço,
# ex.: "não funciona") são procuradas no texto normalizado, porque o "não"
# seria removido como stopword na etapa 4.
ROTEADOR_SETORES = {
    "Financeiro": [
        "pagamento", "fatura", "cobrança", "boleto", "reembolso",
        "cartão", "cancelar", "pix", "assinatura",
    ],
    "Suporte Técnico": [
        "erro", "travou", "trava", "bug", "lento", "instalar",
        "acesso", "login", "não funciona", "não abre", "não consigo",
    ],
}

# Ordem de desempate quando dois setores empatam em número de palavras-chave
# (financeiro primeiro: reclamações de dinheiro costumam ser mais críticas).
ORDEM_SETORES = ["Financeiro", "Suporte Técnico"]

# Pacotes NLTK necessários:
#   punkt      -> tokenização (word_tokenize)
#   punkt_tab  -> dados tabulares do Punkt exigidos pelas versões novas do NLTK
#   stopwords  -> nltk.corpus.stopwords (português)
#   vader_lexicon -> lexicon do VADER (SentimentIntensityAnalyzer, inglês)
_NLTK_PACOTES = ["punkt", "punkt_tab", "stopwords", "vader_lexicon"]


def garantir_dependencias_nltk():
    """Baixa automaticamente os pacotes NLTK na primeira execução.

    É idempotente: quando o pacote já existe, o download não faz nada.
    Se estiver sem internet, os erros são engolidos e os recursos que
    dependem dos dados terão fallbacks seguros (ver _STOPWORDS_PT e VADER).
    """
    import nltk

    for pacote in _NLTK_PACOTES:
        try:
            nltk.download(pacote, quiet=True)
        except Exception as exc:  # rede indisponível etc.
            print(f"[aviso] Não foi possível baixar '{pacote}': {exc}")


# Garante os dados antes de qualquer uso de tokenização/stopwords/VADER.
garantir_dependencias_nltk()

import nltk  # noqa: E402
from nltk.corpus import stopwords  # noqa: E402
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402

# Set de stopwords em português (Atividade 4). Fallback seguro: se o corpus
# não existir, segue com conjunto vazio (nenhuma palavra é removida).
try:
    _STOPWORDS_PT = set(stopwords.words("portuguese"))
except Exception:
    _STOPWORDS_PT = set()

# Analisador VADER (inglês). Fallback seguro: se o lexicon não estiver
# disponível, o sentimento roda apenas pela regra em PT-BR.
try:
    _ANALISADOR_VADER = SentimentIntensityAnalyzer()
except Exception:
    _ANALISADOR_VADER = None

# Stemmer português. Fallback seguro: se o Snowball não estiver disponível,
# `_radical` simplesmente não corta sufixo (só remove acentos), e o
# casamento por radical exato + prefixo continua funcionando.
try:
    from nltk.stem.snowball import SnowballStemmer

    _STEMMER_PT = SnowballStemmer("portuguese")
except Exception:
    _STEMMER_PT = None


# ==============================================================================
# COMPARAÇÃO DE PALAVRAS — TOLERANTE A ACENTO E FLEXÃO
# ==============================================================================
# POR QUE ISTO EXISTE
# O detector de palavras negativas (Atividade 3) e as regras de sentimento/
# categoria comparavam o token com a palavra do vocabulário por IGUALDADE
# EXATA de string. Isso quebrava de duas formas:
#
#   1. ACENTO — o cliente digita "pessimo", "otimo", "reclamacao" (sem acento,
#      o que é comum ao digitar rápido). O vocabulário tem "péssimo",
#      "ótimo", "reclamação" COM acento -> nunca casava.
#   2. FLEXÃO — o vocabulário tem "erro"/"problema"/"péssimo", mas o texto
#      traz "erros"/"problemas"/"pessimidade" -> nunca casava.
#
# Resultado: mensagens claramente críticas ("pessimo, o site ta todo errado
# nao abre") retornavam "Nenhuma palavra crítica detectada".
#
# A SOLUÇÃO
# Antes de comparar, reduzimos as DUAS pontas ao mesmo radical:
#   texto do cliente -> sem acento -> radical (stemmer)
#   palavra do léxico -> sem acento -> radical (stemmer)
# Assim "pessimo", "péssimos" e "péssima" caem todos no mesmo radical "pessim".
#
# IMPORTANTE — os radicais são usados SÓ PARA COMPARAR. Os tokens exibidos,
# o "top palavras" e a normalização continuam com o texto original (com
# acento), então a saída legível para o usuário não muda em nada.
#
# A comparação aceita radical exato OU radical-prefixo (um começa com o
# outro) com no mínimo _RADICAL_MINIMO letras de lado. O piso de 4 letras
# evita falso positivo do tipo "cara" -> radical "car" casando com "carinho".
# ---------------------------------------------------------------------------

# Tamanho mínimo (em letras) para que a comparação por prefixo valha.
_RADICAL_MINIMO = 4


def _sem_acento(texto: str) -> str:
    """Remove os acentos: "péssimo" -> "pessimo", "reclamação" -> "reclamacao".

    A decomposição Unicode NFD separa cada caractere acentuado em letra base
    + marca combinante; basta descartar as marcas (categoria Mn) e voltar
    para a forma normalizada NFC.
    """
    decomposto = unicodedata.normalize("NFD", texto.lower())
    sem_marcas = "".join(
        caractere
        for caractere in decomposto
        if unicodedata.category(caractere) != "Mn"
    )
    return unicodedata.normalize("NFC", sem_marcas)


def _radical(palavra: str) -> str:
    """Reduz uma palavra ao radical de comparação: sem acento e sem flexão.

    "péssimo" -> "pessim" | "demorando" -> "demor" | "erros" -> "erros"
    (o Snowball em português é limitado: "erro" e "erros" às vezes recebem
    radicais diferentes — por isso a comparação também aceita PREFIXO, e não
    só igualdade.)
    """
    limpa = _sem_acento(palavra)
    if _STEMMER_PT is None:
        return limpa
    try:
        return _STEMMER_PT.stem(limpa)
    except Exception:
        return limpa


def _radicalizar_texto(texto: str) -> str:
    """Aplica `_radical` a cada palavra e devolve o texto unido por espaços.

    Usado para casar palavras-chave COM ESPAÇO ("não funciona", "não consigo"),
    que não podem ser procuradas token a token.
    """
    return " ".join(_radical(palavra) for palavra in texto.split())


def _construir_indice(vocabulario):
    """Monta {radical: [termos do léxico]} para busca por radical.

    Vários termos podem cair no mesmo radical, então o índice guarda uma
    lista de termos por radical, e não um único termo.
    """
    indice = {}
    for termo in vocabulario:
        indice.setdefault(_radical(termo), []).append(termo)
    return indice


def _termos_que_casam(indice, radical: str) -> set:
    """Devolve os termos do léxico casados com o radical informado.

    Casa por igualdade exata de radical OU por prefixo mútuo, desde que ambos
    tenham pelo menos _RADICAL_MINIMO letras (evita casar radicais curtos demais
    e gerar falsos positivos).
    """
    if not radical:
        return set()
    achados = set()
    for radical_lexico, termos in indice.items():
        if radical == radical_lexico:
            achados.update(termos)
        elif min(len(radical), len(radical_lexico)) >= _RADICAL_MINIMO and (
            radical.startswith(radical_lexico) or radical_lexico.startswith(radical)
        ):
            achados.update(termos)
    return achados


# Índices de radical de cada vocabulário, prontos para busca. Construídos uma
# única vez na importação: as listas acima são a fonte da verdade, e mudar uma
# palavra aqui reflete no índice automaticamente.
#   {"pessim": ["péssimo"], "ruim": ["ruim"], ...}
_INDICE_NEGATIVAS = _construir_indice(PALAVRAS_NEGATIVAS)
_INDICE_POSITIVAS = _construir_indice(PALAVRAS_POSITIVAS)
# Frases negativas já convertidas para radical de texto, para busca por
# substring no texto normalizado (ver detectar_palavras_negativas).
_FRASES_NEGATIVAS_RADICAL = [
    _radicalizar_texto(frase) for frase in PALAVRAS_NEGATIVAS_FRASES
]
# Roteador: índice por setor, separando termos simples (busca por token) de
# frases (busca por radical de texto, porque "não" cai como stopword na
# etapa 4 e a frase precisa ser procurada no texto já normalizado).
_INDICE_ROTEADOR = {
    setor: {
        "radicais": _construir_indice([termo for termo in termos if " " not in termo]),
        "frases_radical": [_radicalizar_texto(termo) for termo in termos if " " in termo],
    }
    for setor, termos in ROTEADOR_SETORES.items()
}



# ==============================================================================
# ETAPA 1 — LIMPEZA DO TEXTO  [ATIVIDADE 9]
# ==============================================================================
def normalizar_texto(texto: str) -> str:
    """Minúsculas + remoção de pontuação/caracteres especiais + espaços únicos.

    Raciocínio: antes de qualquer análise, reduzimos o texto a palavras puras
    (letras com acento, dígitos) separadas por um espaço — sem pontuação que
    poluiria a tokenização, as contagens e o casamento de palavras-chave.
    Ex.: "Olá, Mundo!! ISSO é um TESTE." -> "olá mundo isso é um teste"
    """
    texto = texto.lower()
    # [\W_] casa todo caractere que NÃO seja palavra (pontuação, símbolos)
    # e também o sublinhado; substitui a sequência inteira por um espaço.
    texto = re.sub(r"[\W_]+", " ", texto)
    # Colapsa espaços múltiplos e remove espaços das bordas.
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# ==============================================================================
# ETAPA 2 — TRADUÇÃO PT -> EN GRATUITA (ponte para recursos do NLTK em inglês)
# ==============================================================================
# Os endpoints públicos do Google recusam requisições anônimas de cliente
# desconhecido: respondem HTTP 429 com uma página HTML "Sorry..." em vez do
# JSON. O que o Google checa é o cabeçalho — simular um navegador com um
# User-Agent real (mais Accept-Language/Referer coerentes) faz o endpoint
# responder 200 normalmente. Estes headers são o requisito que faltava.
_HEADERS_NAVEGADOR = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "Referer": "https://translate.google.com/",
    "Connection": "keep-alive",
}

# Limites de tamanho por provedor (com folga sobre o máximo real do serviço).
LIMITE_CHARS_GOOGLE = 4500    # caracteres aceitos pelo endpoint do Google
LIMITE_BYTES_MYMEMORY = 450   # bytes aceitos pelo MyMemory (máx. real: 500)

# Sessão reaproveitada entre as requisições: mantém cookies e a conexão viva,
# o que reduz a chance de o Google exibir o desafio de verificação.
_SESSAO = requests.Session()
_SESSAO.headers.update(_HEADERS_NAVEGADOR)


def _dividir_texto(texto: str, limite: int):
    """Divide o texto em pedaços de no máximo `limite` caracteres.

    O corte acontece em fronteiras de palavra (nunca no meio de um token), o
    que preserva a qualidade da tradução em textos longos.
    """
    if len(texto) <= limite:
        return [texto]

    pedacos, atual = [], ""
    for palavra in texto.split(" "):
        while len(palavra) > limite:  # palavra isolada maior que o limite
            pedacos.append(palavra[:limite])
            palavra = palavra[limite:]
        if len(atual) + len(palavra) + 1 > limite:
            pedacos.append(atual)
            atual = palavra
        else:
            atual = f"{atual} {palavra}".strip()
    if atual:
        pedacos.append(atual)
    return pedacos


def _traduzir_via_chrome(texto: str) -> str:
    """Provedor principal: endpoint público usado pelo dicionário do Chrome.

    É gratuito, dispensa chave de API e — diferentemente do scraper HTML do
    deep-translator, que recebe 429 com frequência — devolve o JSON direto
    quando a requisição traz os headers de navegador.
    """
    partes = []
    for pedaco in _dividir_texto(texto, LIMITE_CHARS_GOOGLE):
        resposta = _SESSAO.get(
            "https://clients5.google.com/translate_a/t",
            params={
                "client": "dict-chrome-ex", "sl": "pt", "tl": "en", "q": pedaco,
            },
            timeout=10,
        )
        resposta.raise_for_status()
        # Formato da resposta: ["tradução", "original", ...]
        dados = resposta.json()
        if isinstance(dados, list) and dados and isinstance(dados[0], str):
            partes.append(dados[0])
    return " ".join(parte for parte in partes if parte).strip()


def _traduzir_via_endpoint_json(texto: str) -> str:
    """Segundo provedor: endpoint JSON clássico do Google (client=gtx)."""
    partes = []
    for pedaco in _dividir_texto(texto, LIMITE_CHARS_GOOGLE):
        resposta = _SESSAO.get(
            "https://translate.googleapis.com/translate_a/single",
            params={
                "client": "gtx", "sl": "pt", "tl": "en",
                "dt": "t", "q": pedaco,
            },
            timeout=10,
        )
        resposta.raise_for_status()
        dados = resposta.json()
        # Formato: [[["tradução", "original", ...], ...], ...]
        trechos = [parte[0] for parte in dados[0] if parte and parte[0]]
        partes.append("".join(trechos))
    return " ".join(parte for parte in partes if parte).strip()


def _traduzir_via_mymemory(texto: str) -> str:
    """Terceiro provedor: MyMemory, independente do Google e sem chave.

    Existe para o caso de o Google continuar bloqueando este IP: como é um
    serviço distinto, o bloqueio de um não afeta o outro. Tem cota diária
    anônima, por isso fica atrás dos provedores do Google.
    """
    partes = []
    for pedaco in _dividir_texto(texto, LIMITE_BYTES_MYMEMORY):
        resposta = _SESSAO.get(
            "https://api.mymemory.translated.net/get",
            params={"q": pedaco, "langpair": "pt|en"},
            timeout=15,
        )
        resposta.raise_for_status()
        partes.append(resposta.json()["responseData"]["translatedText"])
    return " ".join(parte for parte in partes if parte).strip()


def _traduzir_via_deep_translator(texto: str) -> str:
    """Quarto provedor: deep-translator (GoogleTranslator), já declarado em
    requirements.txt. Mantido como último recurso porque o scraper dele é o
    que mais recebe bloqueio anti-bot (HTTP 429)."""
    from deep_translator import GoogleTranslator

    resultado = GoogleTranslator(source="pt", target="en").translate(texto)
    return (resultado or "").strip()


# Ordem de tentativa: mais confiável -> menos confiável.
_PROVEDORES_TRADUCAO = (
    _traduzir_via_chrome,
    _traduzir_via_endpoint_json,
    _traduzir_via_mymemory,
    _traduzir_via_deep_translator,
)


def traduzir_pt_en(texto_normalizado: str) -> str:
    """Traduz PT -> EN com os provedores gratuitos, sem chave e sem custo.

    Percorre a cadeia de provedores até um devolver texto não vazio, de modo
    que um bloqueio (429) ou uma falha de rede em um endpoint não derruba a
    tradução. Se TODOS falharem, retorna TIPO_TRADUCAO_INDISPONIVEL e o
    pipeline segue intacto com a análise em PT-BR.
    """
    if not texto_normalizado:
        return ""

    for provedor in _PROVEDORES_TRADUCAO:
        try:
            resultado = provedor(texto_normalizado)
            if resultado and resultado.strip():
                return resultado.strip()
        except Exception as exc:  # rede, 429, JSON inválido, cota esgotada
            print(
                f"[aviso] Tradução indisponível em {provedor.__name__}: {exc}. "
                f"Tentando o próximo provedor."
            )

    return TIPO_TRADUCAO_INDISPONIVEL


# ==============================================================================
# ETAPA 3 — TOKENIZAÇÃO  [ATIVIDADE 10]
# ==============================================================================
def tokenizar(texto_normalizado: str):
    """Quebra o texto em tokens (palavras) com nltk.word_tokenize.

    A função word_tokenize(punkt) divide por pontuação/espaços respeitando
    contrações e abreviações — é a base de todas as análises seguintes.
    """
    return nltk.word_tokenize(texto_normalizado, language="portuguese")


# ==============================================================================
# ETAPA 4 — REMOÇÃO DE STOPWORDS  [ATIVIDADE 4]
# ==============================================================================
def remover_stopwords(tokens):
    """Remove palavras de alta frequência e baixo significado ("de", "a", "o",
    "para", "não"...). Raciocínio: stopwords poluem a frequência e o casamento
    de palavras-chave — um "top palavras" cheio de "de"/"o" não revela nada
    sobre o comportamento do cliente.
    """
    return [token for token in tokens if token not in _STOPWORDS_PT]


# ==============================================================================
# ETAPA 5 — FREQUÊNCIA DE PALAVRAS  [ATIVIDADES 2 E 7]
# ==============================================================================
def contar_frequencia(tokens_limpos, top_n: int = TOP_N):
    """Conta a frequência das palavras limpas e devolve o top N.

    Atividade 2: entender padrões de comportamento a partir de avaliações.
    Atividade 7: o frontend pode filtrar o histórico por tipo "reclamacao" e
    recalcular o top palavras apenas sobre reclamações — apoiando melhorias
    de produto. A função é a mesma; o "recorte por tipo" acontece na consulta.
    """
    return Counter(tokens_limpos).most_common(top_n)


# ==============================================================================
# ETAPA 6 — DETECÇÃO DE PALAVRAS NEGATIVAS  [ATIVIDADE 3]
# ==============================================================================
def detectar_palavras_negativas(tokens_limpos, texto_normalizado: str = ""):
    """Sinaliza mensagens prioritárias ao encontrar palavras críticas.

    Raciocínio: se o cliente escreveu "ruim", "péssimo", "erro", "problema"
    ou "demora", a mensagem deve ir para o topo da fila de suporte.

    A comparação é feita por RADICAL (ver seção "COMPARAÇÃO DE PALAVRAS"), o
    que faz "pessimo" (sem acento), "péssimos" e "péssima" casarem com
    "péssimo" e aparecerem no alerta.

    Palavras compostas ("não funciona") são procuradas no texto normalizado
    pelo radical, porque não sobrevivem à tokenização com stopword removida.
    A ordem do retorno segue a constante PALAVRAS_NEGATIVAS (determinística).
    """
    achados = set()
    for token in tokens_limpos:
        achados |= _termos_que_casam(_INDICE_NEGATIVAS, _radical(token))

    radical_texto = _radicalizar_texto(texto_normalizado)
    for indice, frase in enumerate(_FRASES_NEGATIVAS_RADICAL):
        if frase and frase in radical_texto:
            achados.add(PALAVRAS_NEGATIVAS_FRASES[indice])

    return [palavra for palavra in PALAVRAS_NEGATIVAS if palavra in achados] + [
        frase for frase in PALAVRAS_NEGATIVAS_FRASES if frase in achados
    ]


# ==============================================================================
# ETAPA 7 — SENTIMENTO (regra PT + reforço VADER no texto EN)  [ATIVIDADES 5/10]
# ==============================================================================
def classificar_sentimento(tokens_limpos, texto_traduzido: str, texto_normalizado: str = ""):
    """Classifica positivo/negativo/neutro combinando DUAS fontes:

    (a) REGRA CONDICIONAL em PT-BR  — conta palavras positivas vs negativas
        nos tokens (Atividades 5 e 10). Não depende de tradução.
    (b) VADER (NLTK) sobre o TEXTO TRADUZIDO PARA INGLÊS — segunda opinião
        e score de reforço. O VADER é excelente em inglês; por isso o texto
        foi traduzido na Etapa 2. Se a tradução falhou, fica "indisponível".

    A contagem de negativas reaproveita `detectar_palavras_negativas`, de modo
    que o alerta de "palavra crítica" e o sentimento NUNCA se contradizem:
    se o alerta disparou, a regra já sabe que o texto é negativo.

    Resultado final (condicional simples combinando as fontes):
        - tradução indisponível ......... prevalece a regra PT-BR;
        - regra neutra .................. o VADER decide;
        - VADER neutro .................. a regra decide;
        - fontes concordam .............. a opinião comum vence;
        - fontes discordam (pos x neg) .. final = neutro (indefinido).
    """
    positivas = sum(
        1 for t in tokens_limpos if _termos_que_casam(_INDICE_POSITIVAS, _radical(t))
    )
    negativas = len(detectar_palavras_negativas(tokens_limpos, texto_normalizado))

    # (a) Regra condicional simples em PT-BR.
    if positivas > negativas:
        regra = "positivo"
    elif negativas > positivas:
        regra = "negativo"
    else:
        regra = "neutro"

    # (b) VADER como reforço, somente se houver tradução disponível.
    scores = None
    if (
        _ANALISADOR_VADER is not None
        and texto_traduzido
        and texto_traduzido != TIPO_TRADUCAO_INDISPONIVEL
    ):
        scores = _ANALISADOR_VADER.polarity_scores(texto_traduzido)
        compound = scores["compound"]
        # Limiares do VADER: >= +0.05 positivo, <= -0.05 negativo, senão neutro.
        if compound >= 0.05:
            vader = "positivo"
        elif compound <= -0.05:
            vader = "negativo"
        else:
            vader = "neutro"
    else:
        vader = "indisponível"

    # Condicional simples combinando as duas fontes.
    if vader == "indisponível":
        final = regra
    elif regra == "neutro":
        final = vader
    elif vader == "neutro":
        final = regra
    elif regra == vader:
        final = regra
    else:
        final = "neutro"  # fontes discordantes -> indefinido

    return {
        "scores_regra": {"positivas": positivas, "negativas": negativas},
        "regra": regra,
        "score_vader": scores,
        "vader": vader,
        "final": final,
    }


# ==============================================================================
# ETAPA 8 — CATEGORIA/SETOR POR PALAVRAS-CHAVE  [ATIVIDADES 6 E 8]
# ==============================================================================
def classificar_categoria(tokens_limpos, texto_normalizado: str) -> str:
    """Direciona a mensagem ao setor correto (chatbot / triagem).

    Regra condicional:
      1. Para cada setor (Financeiro, Suporte Técnico), conta quantas
         palavras-chave aparecem. Palavras simples são procuradas nos tokens
         limpos; frases ("não funciona") no texto normalizado.
      2. Vence o setor com MAIS ocorrências; empate respeita ORDEM_SETORES.
      3. Nenhuma ocorrência -> "Geral".

    Ambos os caminhos comparam por RADICAL, então "cobrancas"/"fatura" e
    "nao funciona" (sem acento) também acionam o roteamento.
    """
    ocorrencias = {}
    radical_texto = _radicalizar_texto(texto_normalizado)
    for setor, indice in _INDICE_ROTEADOR.items():
        total = 0
        for token in tokens_limpos:
            if _termos_que_casam(indice["radicais"], _radical(token)):
                total += 1
        for frase in indice["frases_radical"]:
            if frase in radical_texto:  # frase -> busca no texto normalizado
                total += 1
        ocorrencias[setor] = total

    melhor = max(ocorrencias.values(), default=0)
    if melhor == 0:
        return "Geral"

    candidatos = [s for s, n in ocorrencias.items() if n == melhor]
    for setor in ORDEM_SETORES:
        if setor in candidatos:
            return setor
    return candidatos[0]


# ==============================================================================
# PERSISTÊNCIA — SQLite
# ==============================================================================
# Schema do banco (documentado):
#   id               INTEGER PK — chave primária
#   texto_original   TEXT  — texto digitado pelo usuário (PT-BR), exibição
#   tipo_texto       TEXT  — avaliacao | reclamacao | mensagem
#   texto_normalizado TEXT — texto limpo (minúsculas, sem pontuação)
#   texto_traduzido  TEXT  — tradução PT->EN (ponte p/ VADER); "tradução
#                            indisponível" ou NULL quando a tradução falhou
#   tokens           TEXT  — tokens em PT-BR (JSON list)
#   top_palavras     TEXT  — top N palavras (JSON de [palavra, freq])
#   palavras_negativas TEXT — palavras críticas detectadas (JSON list)
#   sentimento_regra TEXT  — sentimento pela regra PT-BR
#   score_vader      TEXT  — scores do VADER sobre o texto EN (JSON) ou NULL
#   sentimento_final TEXT  — resultado combinado (positivo/negativo/neutro)
#   categoria        TEXT  — Financeiro | Suporte Técnico | Geral
#   criado_em        TEXT  — timestamp ISO (UTC) da análise
SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS analises (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    texto_original     TEXT NOT NULL,
    tipo_texto         TEXT NOT NULL,
    texto_normalizado  TEXT NOT NULL,
    texto_traduzido    TEXT,
    tokens             TEXT NOT NULL,
    top_palavras       TEXT NOT NULL,
    palavras_negativas TEXT NOT NULL,
    sentimento_regra   TEXT NOT NULL,
    score_vader        TEXT,
    sentimento_final   TEXT NOT NULL,
    categoria          TEXT NOT NULL,
    criado_em          TEXT NOT NULL
);
"""


def criar_banco():
    """Cria a tabela na primeira execução (CREATE TABLE IF NOT EXISTS)."""
    with sqlite3.connect(DB_PATH) as conexao:
        conexao.execute(SQL_SCHEMA)


def salvar_no_banco(dados: dict) -> int:
    """Persiste uma análise completa e devolve o id da nova linha."""
    criar_banco()
    with sqlite3.connect(DB_PATH) as conexao:
        cursor = conexao.execute(
            """
            INSERT INTO analises (
                texto_original, tipo_texto, texto_normalizado, texto_traduzido,
                tokens, top_palavras, palavras_negativas, sentimento_regra,
                score_vader, sentimento_final, categoria, criado_em
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dados["texto_original"],
                dados["tipo_texto"],
                dados["texto_normalizado"],
                dados["texto_traduzido"] or None,
                json.dumps(dados["tokens"], ensure_ascii=False),
                json.dumps(dados["top_palavras"], ensure_ascii=False),
                json.dumps(dados["palavras_negativas"], ensure_ascii=False),
                dados["sentimento_regra"],
                json.dumps(dados["score_vader"]) if dados["score_vader"] else None,
                dados["sentimento_final"],
                dados["categoria"],
                dados["criado_em"],
            ),
        )
        return cursor.lastrowid


def _linha_para_dict(linha):
    """Converte uma linha do SELECT em dict (decodificando os campos JSON)."""
    return {
        "id": linha[0],
        "texto_original": linha[1],
        "tipo_texto": linha[2],
        "texto_normalizado": linha[3],
        "texto_traduzido": linha[4],
        "tokens": json.loads(linha[5]),
        "top_palavras": json.loads(linha[6]),
        "palavras_negativas": json.loads(linha[7]),
        "sentimento_regra": linha[8],
        "score_vader": json.loads(linha[9]) if linha[9] else None,
        "sentimento_final": linha[10],
        "categoria": linha[11],
        "criado_em": linha[12],
    }


def listar_historico(where_clause="", params=()):
    """Lista as análises salvas (mais recentes primeiro), com filtros opcionais."""
    criar_banco()
    sql = (
        "SELECT id, texto_original, tipo_texto, texto_normalizado, "
        "texto_traduzido, tokens, top_palavras, palavras_negativas, "
        "sentimento_regra, score_vader, sentimento_final, categoria, criado_em "
        "FROM analises"
    )
    if where_clause:
        sql += f" WHERE {where_clause}"
    sql += " ORDER BY id DESC"

    with sqlite3.connect(DB_PATH) as conexao:
        linhas = conexao.execute(sql, params).fetchall()
    return [_linha_para_dict(l) for l in linhas]


# ==============================================================================
# ORQUESTRADORA — roda TODO o pipeline em ordem e persiste
# ==============================================================================
def analisar_texto(texto: str, tipo_texto: str = "avaliacao") -> dict:
    """Função única que executa todas as etapas para qualquer texto inserido.

    Estilo pedido: A função orquestradora reúne as funções puras de cada
    etapa (normalizar, traduzir, tokenizar, remover_stopwords, contar_frequencia,
    detectar_palavras_negativas, classificar_sentimento, classificar_categoria,
    salvar_no_banco) — permitindo testar e explicar cada etapa isoladamente.
    """
    # --- Validação de entrada ------------------------------------------------
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("O texto não pode estar vazio.")
    if tipo_texto not in TIPOS_VALIDOS:
        raise ValueError(
            f"tipo_texto inválido ('{tipo_texto}'). Use um de: "
            f"{sorted(TIPOS_VALIDOS)}."
        )

    # 1. Limpeza (Atividade 9)
    normalizado = normalizar_texto(texto)
    if len(normalizado) < MIN_TAMANHO_TEXTO:
        raise ValueError(
            "Texto muito curto para análise (mínimo de "
            f"{MIN_TAMANHO_TEXTO} caracteres após a limpeza)."
        )

    # 2. Tradução gratuita PT -> EN (ponte p/ VADER; nunca quebra o pipeline)
    traduzido = traduzir_pt_en(normalizado)

    # 3. Tokenização (Atividade 10)
    tokens = tokenizar(normalizado)

    # 4. Stopwords (Atividade 4)
    tokens_limpos = remover_stopwords(tokens)

    # 5. Frequência (Atividades 2 e 7)
    top = contar_frequencia(tokens_limpos)
    top_palavras = [list(item) for item in top]  # JSON-friendly

    # 6. Palavras negativas (Atividade 3)
    negativas = detectar_palavras_negativas(tokens_limpos, normalizado)

    # 7. Sentimento combinado (Atividades 5 e 10)
    sentimento = classificar_sentimento(tokens_limpos, traduzido, normalizado)

    # 8. Categoria/setor (Atividades 6 e 8)
    categoria = classificar_categoria(tokens_limpos, normalizado)

    # 9. Montagem + persistência
    dados = {
        "texto_original": texto.strip(),
        "tipo_texto": tipo_texto,
        "texto_normalizado": normalizado,
        "texto_traduzido": traduzido,
        "tokens": tokens_limpos,
        "top_palavras": top_palavras,
        "palavras_negativas": negativas,
        "sentimento_regra": sentimento["regra"],
        "score_vader": sentimento["score_vader"],
        "sentimento_final": sentimento["final"],
        "categoria": categoria,
        "criado_em": datetime.now(timezone.utc).isoformat(),
    }
    dados["id"] = salvar_no_banco(dados)
    return dados


# ==============================================================================
# APLICAÇÃO FLASK (backend da interface)
# ==============================================================================
app = Flask(__name__)
app.json.ensure_ascii = False  # respostas JSON com acentos legíveis


@app.get("/")
def rota_index():
    """Serve o frontend único (templates/index.html)."""
    return render_template("index.html")


@app.post("/analisar")
def rota_analisar():
    """REST: analisa um texto. Corpo JSON: {texto, tipo_texto}."""
    corpo = request.get_json(silent=True) or {}
    texto = corpo.get("texto", "")
    tipo_texto = corpo.get("tipo_texto", "avaliacao")

    try:
        resultado = analisar_texto(texto, tipo_texto)
    except ValueError as erro:  # texto vazio/curto/tipo inválido
        return jsonify({"erro": str(erro)}), 400
    except sqlite3.Error as erro:  # falha de banco
        return jsonify({"erro": f"Falha ao salvar no banco: {erro}"}), 500
    return jsonify(resultado), 200


@app.get("/historico")
def rota_historico():
    """REST: lista o histórico. Query params opcionais:
    ?sentimento=negativo  ?categoria=Financeiro  ?tipo=reclamacao
    """
    sentimento = request.args.get("sentimento", "").strip()
    categoria = request.args.get("categoria", "").strip()
    tipo = request.args.get("tipo", "").strip()

    clausulas, params = [], []
    if sentimento:
        clausulas.append("sentimento_final = ?")
        params.append(sentimento)
    if categoria:
        clausulas.append("categoria = ?")
        params.append(categoria)
    if tipo:
        clausulas.append("tipo_texto = ?")
        params.append(tipo)

    where = (" AND ".join(clausulas)) if clausulas else ""
    try:
        registros = listar_historico(where, params)
    except sqlite3.Error as erro:
        return jsonify({"erro": f"Falha ao consultar o banco: {erro}"}), 500
    return jsonify(registros), 200


if __name__ == "__main__":
    criar_banco()  # garante o schema na primeira execução
    porta = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    print(f"Pipelines de PLN rodando em http://127.0.0.1:{porta}")
    app.run(host="127.0.0.1", port=porta, debug=debug)
