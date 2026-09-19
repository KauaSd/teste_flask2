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

TRADUÇÃO SEM CHAVE E SEM CUSTO:
O deep-translator, com o backend GoogleTranslator, acessa o endpoint público
e gratuito de tradução do Google — NÃO usa o Google Cloud Translate API
(serviço pago, que exige credenciais/API key). Aqui a tradução não gera
custo nem precisa de chave. Há ainda um fallback leve que acessa o mesmo
endpoint público gratuito direto via `requests`, caso o deep-translator
esteja bloqueado (bastante comum: esse IP pode receber HTTP 429 do scraper
do deep-translator, enquanto o endpoint JSON continua acessível).
Se ambos falharem (ex.: sem internet), o pipeline NÃO quebra: segue com a
análise 100% em PT-BR e registra "tradução indisponível".
================================================================================
"""

import json
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime, timezone

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
PALAVRAS_NEGATIVAS = [
    "ruim", "péssimo", "erro", "problema", "demora", "demorou",
    "falha", "defeito", "horrível", "atraso", "atrasou", "lento",
    "travou", "trava", "quebrou", "insatisfeito", "reclamação", "cara",
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
def _traduzir_google_gratuito_direto(texto: str) -> str:
    """Fallback leve para o MESMO serviço gratuito do Google.

    O deep-translator scraper (HTML) recebe HTTP 429 em alguns IPs,
    enquanto o endpoint público JSON continua acessível. Este fallback usa o
    endpoint translate.googleapis.com/translate_a/single — gratuito, SEM chave
    de API e SEM custo, igual ao backend GoogleTranslator do deep-translator.
    Retorna TIPO_TRADUCAO_INDISPONIVEL se a rede/endpoint falharem.
    """
    try:
        import requests

        resposta = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={
                "client": "gtx", "sl": "pt", "tl": "en",
                "dt": "t", "q": texto,
            },
            timeout=10,
        )
        resposta.raise_for_status()
        dados = resposta.json()
        # A resposta tem o formato [[["tradução", "original", ...], ...], ...]
        trechos = [parte[0] for parte in dados[0] if parte and parte[0]]
        traducao = "".join(trechos).strip()
        return traducao or TIPO_TRADUCAO_INDISPONIVEL
    except Exception:
        return TIPO_TRADUCAO_INDISPONIVEL


def traduzir_pt_en(texto_normalizado: str) -> str:
    """Traduz PT -> EN usando deep-translator (GoogleTranslator gratuito).

    - NÃO usa chave de API nem gera custo (diferente do Google Cloud
      Translate API, que é pago e exige credenciais).
    - O try/except externo garante que falhas de rede, bloqueios (429) ou
      indisponibilidade do serviço NÃO quebram o pipeline.
    - Se a tradução falhar, retorna TIPO_TRADUCAO_INDISPONIVEL e o pipeline
      segue apenas com a análise em PT-BR.
    """
    if not texto_normalizado:
        return ""

    # Opção principal: deep-translator (biblioteca gratuita, sem chave).
    try:
        from deep_translator import GoogleTranslator

        tradutor = GoogleTranslator(source="pt", target="en")
        resultado = tradutor.translate(texto_normalizado)
        if resultado and resultado.strip():
            return resultado.strip()
    except Exception:
        pass  # cai no fallback abaixo

    # Fallback: mesmo serviço gratuito do Google, via requests direto.
    return _traduzir_google_gratuito_direto(texto_normalizado)


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
def detectar_palavras_negativas(tokens_limpos):
    """Sinaliza mensagens prioritárias ao encontrar palavras críticas.

    Raciocínio: se o cliente escreveu "ruim", "péssimo", "erro", "problema"
    ou "demora", a mensagem deve ir para o topo da fila de suporte.
    A ordem do retorno segue a constante PALAVRAS_NEGATIVAS (determinística).
    """
    presentes = set(tokens_limpos)
    return [palavra for palavra in PALAVRAS_NEGATIVAS if palavra in presentes]


# ==============================================================================
# ETAPA 7 — SENTIMENTO (regra PT + reforço VADER no texto EN)  [ATIVIDADES 5/10]
# ==============================================================================
def classificar_sentimento(tokens_limpos, texto_traduzido: str):
    """Classifica positivo/negativo/neutro combinando DUAS fontes:

    (a) REGRA CONDICIONAL em PT-BR  — conta palavras positivas vs negativas
        nos tokens (Atividades 5 e 10). Não depende de tradução.
    (b) VADER (NLTK) sobre o TEXTO TRADUZIDO PARA INGLÊS — segunda opinião
        e score de reforço. O VADER é excelente em inglês; por isso o texto
        foi traduzido na Etapa 2. Se a tradução falhou, fica "indisponível".

    Resultado final (condicional simples combinando as fontes):
        - tradução indisponível ......... prevalece a regra PT-BR;
        - regra neutra .................. o VADER decide;
        - VADER neutro .................. a regra decide;
        - fontes concordam .............. a opinião comum vence;
        - fontes discordam (pos x neg) .. final = neutro (indefinido).
    """
    positivas = sum(1 for t in tokens_limpos if t in PALAVRAS_POSITIVAS)
    negativas = sum(1 for t in tokens_limpos if t in PALAVRAS_NEGATIVAS)

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
    """
    tokens = set(tokens_limpos)
    ocorrencias = {}
    for setor, palavras_chave in ROTEADOR_SETORES.items():
        total = 0
        for palavra_chave in palavras_chave:
            if " " in palavra_chave:  # frase -> busca no texto normalizado
                if palavra_chave in texto_normalizado:
                    total += 1
            elif palavra_chave in tokens:  # palavra simples -> nos tokens
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
    negativas = detectar_palavras_negativas(tokens_limpos)

    # 7. Sentimento combinado (Atividades 5 e 10)
    sentimento = classificar_sentimento(tokens_limpos, traduzido)

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