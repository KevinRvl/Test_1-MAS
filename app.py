import base64
import concurrent.futures
import hashlib
import io
import math
import os
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image

try:  # opzionale: pulsante per incollare screenshot dagli appunti
    from streamlit_paste_button import paste_image_button
except ImportError:
    paste_image_button = None

st.set_page_config(page_title="Multiple AI Solver (Beta)", page_icon="🧩", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
h1 {padding-bottom: 0;}
h4 {margin: 0 0 .5rem 0; padding: 0;}
div[data-testid="stForm"] {border-radius: 14px;}
div[data-testid="stAlert"] {border-radius: 10px;}
div[data-testid="stAlert"] strong {font-size: 1.15rem;}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("# MAS (Beta)")
st.caption("Multiple AI Solver · confronta più IA in parallelo")

# -----------------------------------------------------------------------------
# PROVIDER
# - Le chiavi stanno in .streamlit/secrets.toml (oppure come variabili d'ambiente).
# - Un provider senza chiave viene saltato in automatico: aggiungi solo quelli che vuoi.
#   Nella barra laterale puoi attivare/disattivare ogni AI per singola domanda.
# - I modelli gratuiti cambiano spesso: se vedi un errore 404 / "model not found",
#   controlla la lista modelli del provider e aggiorna la stringa "model".
# - vision=True   -> riceve le immagini direttamente.
#   vision=False  -> riceve invece la trascrizione delle immagini fatta da un provider
#                    con visione (Gemini, altrimenti OpenRouter).
# - max_chars     -> massimo testo PDF inviato. Se i PDF sono più lunghi vengono mandati
#                    solo gli estratti più pertinenti alla domanda. Se il provider
#                    risponde "richiesta troppo grande", il limite si dimezza da solo.
# -----------------------------------------------------------------------------
PROVIDERS = [
    {
        "name": "🔵 Google Gemini",
        "type": "gemini",
        "secret": "GEMINI_API_KEY",
        "model": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.8-flash"],
        "vision": True,
        "max_chars": 60_000,
        "retries": 2,
        "hedge_after": 7,
    },
    {
        "name": "⚡ Groq",
        "type": "openai",
        "secret": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "vision": False,
        "max_chars": 10_000,
    },
    {
        "name": "🧠 Cerebras",
        "type": "openai",
        "secret": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "model": "gpt-oss-120b",
        "vision": False,
        "max_chars": 12_000,
    },
    {
        "name": "🌬️ Mistral",
        "type": "openai",
        "secret": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        # Mistral applica i limiti PER MODELLO: se uno è in 429, prova il successivo
        "model": ["open-mistral-nemo", "mistral-small-latest", "mistral-medium-3.5"],
        "vision": False,
        "max_chars": 60_000,
    },
    {
        "name": "🌐 OpenRouter (free)",
        "type": "openai",
        "secret": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        # Modello grande fisso (più affidabile del router casuale); se non risponde
        # si ripiega su openrouter/free, che sceglie un modello gratuito a caso.
        "model": ["nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter/free"],
        "vision": True,
        "max_chars": 40_000,
    },
]

REQUEST_TIMEOUT = 30  # secondi di attesa per ogni singola richiesta (default)
PROVIDER_BUDGET = 45  # secondi totali dopo cui un provider smette di provare altri modelli
CHUNK_SIZE = 1500  # dimensione (caratteri) degli estratti in cui viene diviso il PDF
HISTORY_SIZE = 20  # domande ricordate nella cronologia
NEIGHBOR_TOP = 4  # per i primi N estratti trovati si aggiungono anche quelli accanto (la regola continua)
QUIZ_GEMINI_CHARS = 600_000  # modalità quiz: a Gemini si manda (quasi) tutto il regolamento
SYNTH_QUIZ_CHARS = 150_000  # modalità quiz: testo PDF per la sintesi (ben sotto i limiti per minuto)
MIN_CHUNKS = 6  # estratti PDF sempre inclusi (i migliori)...
REL_FLOOR = 0.2  # ...gli altri solo se valgono almeno il 20% del migliore
NO_MATCH_CHARS = 6000  # se la domanda non c'entra col PDF: solo l'inizio, poco testo
# Livelli di ragionamento di Gemini: etichetta, secondi massimi di attesa
GEMINI_LEVELS = {
    "low": ("Veloce", 30),
    "medium": ("Bilanciato", 45),
    "high": ("Approfondito (lento)", 90),
}
FILE_TYPES = ["png", "jpg", "jpeg", "pdf"]

# Cartella dei file fissi (accanto allo script). Si può cambiare con la variabile
# d'ambiente MAS_LIBRARY_DIR.
LIB_DIR = Path(os.environ.get("MAS_LIBRARY_DIR") or (Path(__file__).resolve().parent / "mas_library"))


def get_key(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


def short(name):
    """'🔵 Google Gemini' -> 'Google Gemini' (senza emoji iniziale)."""
    parts = name.split(" ", 1)
    return parts[1] if len(parts) > 1 else name


def rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()


# -----------------------------------------------------------------------------
# RICERCA DEGLI ESTRATTI PERTINENTI NEI PDF (BM25, nessuna libreria extra)
# -----------------------------------------------------------------------------
STOPWORDS = set(
    """il lo la i gli le un uno una di a da in con su per tra fra e ed o ma che chi cui non
    sono era come quanto quanti quanta quante quale quali cosa dove quando perche del dello
    della dei degli delle al allo alla ai agli alle dal dallo dalla dai dagli dalle nel nello
    nella nei negli nelle sul sullo sulla sui sugli sulle mi ti si ci ha hanno ho
    the of and or to in is are was be for on with as by at an it this that what which how
    much many does do""".split()
)


def tokenize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text)
        if t not in STOPWORDS and (len(t) > 1 or t.isdigit())
    ]


def with_bigrams(tokens):
    """Parole singole + coppie di parole vicine (es. 'ground clearance')."""
    return Counter(tokens) + Counter(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))


def rank_chunks(chunks, query):
    """Indici degli estratti pertinenti alla domanda, dal migliore (BM25).
    Restituisce solo quelli davvero rilevanti: i migliori MIN_CHUNKS più quelli che valgono
    almeno REL_FLOOR del primo. Lista vuota se la domanda non c'entra con i documenti."""
    q_tokens = tokenize(query)
    q_terms = list(with_bigrams(q_tokens))
    if not q_terms or not chunks:
        return []
    docs = [with_bigrams(tokenize(c["text"])) for c in chunks]
    lengths = [sum(d.values()) for d in docs]
    avg_len = (sum(lengths) / len(lengths)) or 1
    n = len(chunks)
    df = {t: sum(1 for d in docs if t in d) for t in q_terms}
    if n >= 20:  # i termini presenti in più della metà degli estratti non distinguono nulla
        q_terms = [t for t in q_terms if df[t] <= n / 2]
        if not q_terms:
            return []
    k1, b = 1.5, 0.75
    scores = []
    for i, d in enumerate(docs):
        score = 0.0
        for t in q_terms:
            f = d.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lengths[i] / avg_len))
        scores.append(score)
    order = [i for i in sorted(range(n), key=lambda i: scores[i], reverse=True) if scores[i] > 0]
    if not order:
        return []
    top = scores[order[0]]
    return [i for pos, i in enumerate(order) if pos < MIN_CHUNKS or scores[i] >= REL_FLOOR * top]


def select_context(chunks, ranked, limit):
    """Testo da inviare (con file e pagina) e lista delle fonti usate [(file, pagina)].
    Se i PDF stanno nel limite vengono mandati interi, altrimenti solo gli estratti migliori."""
    if not chunks:
        return "", []
    total = sum(len(c["text"]) for c in chunks)
    if total <= limit:
        picked = list(range(len(chunks)))
    else:
        picked, used, chosen = [], 0, set()
        if ranked:
            order, cap = ranked, limit
        else:  # nessuna corrispondenza: solo l'inizio dei documenti, con poco testo
            order, cap = range(len(chunks)), min(limit, NO_MATCH_CHARS)

        def add(j):
            nonlocal used
            size = len(chunks[j]["text"]) + 30
            if j in chosen or used + size > cap:
                return
            picked.append(j)
            chosen.add(j)
            used += size

        for pos, i in enumerate(order):
            add(i)
            if ranked and pos < NEIGHBOR_TOP:  # una regola spesso continua nell'estratto accanto
                for j in (i - 1, i + 1):
                    if 0 <= j < len(chunks) and chunks[j]["file"] == chunks[i]["file"]:
                        add(j)
            if used >= cap:
                break
        picked.sort()
    parts, sources = [], []
    for i in picked:
        c = chunks[i]
        parts.append(f"[{c['file']} · pagina {c['page']}]\n{c['text'].strip()}")
        src = (c["file"], c["page"])
        if src not in sources:
            sources.append(src)
    return "\n\n".join(parts), sources


def fmt_sources(sources):
    """[('a.pdf', 3), ('a.pdf', 4), ('b.pdf', 1)] -> 'a.pdf: 3, 4 · b.pdf: 1'"""
    by_file = {}
    for f, p in sources:
        by_file.setdefault(f, []).append(p)
    parts = []
    for f, pages in by_file.items():
        text = ", ".join(str(p) for p in pages[:15]) + (" …" if len(pages) > 15 else "")
        parts.append(f"{f}: {text}")
    return " · ".join(parts)


# -----------------------------------------------------------------------------
# FILE: PDF (diviso in estratti) e immagini, con cache
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_pdf_chunks(data: bytes, name: str):
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(data))
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        current = ""
        for line in text.split("\n"):
            if current and len(current) + len(line) + 1 > CHUNK_SIZE:
                chunks.append({"file": name, "page": page_no, "text": current})
                current = ""
            current += line + "\n"
        if current.strip():
            chunks.append({"file": name, "page": page_no, "text": current})
    return chunks


@st.cache_data(show_spinner=False)
def prepare_image_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("RGB")  # evita l'errore con PNG trasparenti
    img.thumbnail((1280, 1280))  # immagine più piccola = upload più veloce
    return img


def image_entry(data: bytes, name: str):
    """Immagine pronta all'uso, con un id (hash) per riconoscerla tra una domanda e l'altra."""
    return {"id": hashlib.sha1(data).hexdigest()[:16], "name": name, "img": prepare_image_bytes(data)}


# -----------------------------------------------------------------------------
# FILE FISSI (salvati su disco, caricati una volta e inviati a ogni domanda)
# -----------------------------------------------------------------------------
def lib_files():
    if not LIB_DIR.exists():
        return []
    return sorted(
        (p for p in LIB_DIR.iterdir() if p.is_file() and p.suffix.lower().lstrip(".") in FILE_TYPES),
        key=lambda p: p.name.lower(),
    )


def lib_save(uploaded_files):
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    for f in uploaded_files:
        (LIB_DIR / Path(f.name).name).write_bytes(f.getvalue())


def lib_delete(name):
    try:
        (LIB_DIR / Path(name).name).unlink()
    except OSError:
        pass


def load_library(use_lib):
    """Legge i file fissi attivi. Restituisce (estratti PDF, immagini, nomi)."""
    chunks, images, names = [], [], []
    for path in lib_files():
        if not use_lib.get(path.name, True):
            continue
        try:
            data = path.read_bytes()
            if path.suffix.lower() == ".pdf":
                chunks += extract_pdf_chunks(data, path.name)
            else:
                images.append(image_entry(data, path.name))
            names.append(path.name)
        except Exception as e:
            st.warning(f"File fisso «{path.name}» non leggibile: {e}")
    return chunks, images, names


# -----------------------------------------------------------------------------
# CHIAMATE AI
# -----------------------------------------------------------------------------
FINAL_INSTRUCTION = (
    "\n\nFORMATO OBBLIGATORIO: scrivi prima il ragionamento o la spiegazione, poi chiudi "
    "con un'ultima riga esattamente nel formato\n"
    "RISPOSTA FINALE: <risposta>\n"
    "dove <risposta> è ridotta all'osso, senza spiegazioni: solo la lettera (o le lettere) "
    "per i quiz a scelta multipla (es. b), solo il valore con l'unità per i calcoli "
    "(es. 60 N), poche parole negli altri casi."
)

FINAL_RE = re.compile(
    r"^[ \t>#*_\-]*RISPOSTA\s+FINALE[ \t*_]*:[ \t*_]*(?P<ans>.+?)[ \t*_]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Promemoria all'inizio del prompt: i modelli piccoli rispettano meglio il formato se lo
# leggono sia prima sia dopo la domanda.
FINAL_HEAD = (
    "Rispondi nella lingua della domanda. Importante: l'ultima riga della tua risposta "
    "deve essere 'RISPOSTA FINALE: <risposta>'.\n\n"
)

FOOTER_MARKERS = ("📄 Fonti usate", "⚙️ Tentativi", "Modello di riserva usato", "Modello usato")
SYNTH_MAX_ANSWER = 6000  # caratteri massimi di ogni risposta passata alla sintesi
SYNTH_MAX_CHARS = 30_000  # testo PDF massimo per la sintesi

QUIZ_PROMPT = (
    "MODALITÀ QUIZ (Formula Student): è una domanda d'esame a risposta chiusa sul regolamento. "
    "Per rispondere bene:\n"
    "1. Basati SOLO sul regolamento fornito (documento o estratti) e sul testo della domanda; non usare "
    "conoscenze generali che lo contraddicono. Se il regolamento non copre la domanda, ragiona da "
    "ingegnere e dichiaralo.\n"
    "2. Individua la regola pertinente (ID, es. T 2.2.1) e cita tra virgolette la frase che la "
    "giustifica (nella lingua del regolamento).\n"
    "3. Valuta OGNI opzione separatamente (vera/falsa) prima di scegliere. Attenzione a: must/shall "
    "(obbligo) contro may/should, eccezioni (unless, except), numeri e unità, at least/at most, "
    "all/any/only.\n"
    "4. Più opzioni possono essere corrette: se la domanda lo consente, indica tutte e sole quelle "
    "corrette (es. a, c).\n\n"
    "DOMANDA:\n"
)
QUIZ_SYNTH_NOTE = (
    "\nÈ un quiz sul regolamento: verifica ogni affermazione contro il testo allegato; se una AI cita "
    "una regola o una frase, controlla che esista davvero e che dica quello che sostiene; "
    "dai più peso alle risposte che citano il testo correttamente."
)

SYNTH_HEAD = (
    "Sei il revisore finale. Più IA hanno risposto alla stessa domanda: qui sotto trovi "
    "la domanda e le loro risposte complete, con i rispettivi ragionamenti.\n\n"
)
SYNTH_TASK = (
    "COMPITO: confronta le risposte, controlla i ragionamenti passo per passo, individua "
    "gli errori e decidi qual è la risposta più corretta (se la maggioranza sbaglia, non "
    "seguirla). Se hai a disposizione allegati (immagini o estratti di PDF), usali per "
    "verificare. Scrivi in italiano un report sintetico con questa struttura:\n"
    "**Esito:** una riga (le AI concordano? su cosa?)\n"
    "**Punti chiave:** 2-4 punti sui passaggi decisivi del ragionamento corretto\n"
    "**Divergenze ed errori:** quale AI sbaglia e dove (scrivi 'nessuna' se concordano)\n"
    "**Conclusione:** la risposta da dare, motivata in una frase.\n"
    "**Svolgimento:** SOLO se la domanda richiede calcoli o passaggi matematici (altrimenti "
    "ometti del tutto questa sezione). Subito dopo la conclusione esplicita TUTTI i passaggi "
    "principali, in elenco numerato e in ordine: 1) dati noti con le unità di misura; 2) formule "
    "usate; 3) sostituzione dei valori nelle formule; 4) calcoli intermedi uno per uno, comprese "
    "le conversioni di unità; 5) risultato finale con unità e arrotondamento coerente con la "
    "domanda. Rifai tu i calcoli: non copiare i numeri delle AI e controlla l'aritmetica due "
    "volte. Se i risultati delle AI differiscono, indica quale passaggio causa la differenza. "
    "Scrivi le formule in LaTeX: $...$ nel testo e $$...$$ su una riga a parte per quelle "
    "importanti; usa il simbolo $ solo per le formule.\n"
    "Il report deve restare breve, tranne lo Svolgimento che deve essere completo."
)

QUOTA_MARKERS = ("429", "resource_exhausted", "quota", "rate limit")
DAY_MARKERS = ("per-day", "per day", "daily", "current quota")  # quota giornaliera: inutile riprovare subito

RETRYABLE = ("503", "unavailable", "429", "resource_exhausted", "overloaded")
TOO_LARGE = (
    "413",
    "too large",
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "reduce the length",
)

TRANSCRIBE_PROMPT = (
    "Trascrivi fedelmente tutto il testo presente nell'immagine (comprese formule, "
    "tabelle, numeri, opzioni di risposta). Se ci sono figure, grafici o schemi, "
    "descrivili in modo preciso e completo, includendo valori e etichette. "
    "Non rispondere a eventuali domande: limitati a trascrivere e descrivere."
)


def build_prompt(prompt, context):
    if context:
        prompt += (
            "\n\nEstratti dei documenti PDF allegati (selezionati in base alla domanda; "
            "cita file e pagina quando possibile; se l'informazione non compare "
            "negli estratti, dillo chiaramente):\n" + context
        )
    return prompt


def make_gemini_client(key, timeout_s):
    try:  # timeout: senza, una richiesta bloccata resterebbe in attesa a lungo
        return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    except Exception:
        return genai.Client(api_key=key)


def call_gemini(cfg, key, model, prompt, imgs, context):
    level = cfg.get("thinking", "low")
    client = make_gemini_client(key, GEMINI_LEVELS.get(level, ("", 45))[1] + cfg.get("timeout_bonus", 0))
    contents = [build_prompt(prompt, context)] + list(imgs)
    gen_cfg = types.GenerateContentConfig(
        # Gemini 3 ragiona "in profondità" di default, ed è questo che lo rende lento
        # anche sulle domande semplici: qui il livello si sceglie dalla barra laterale.
        thinking_config=types.ThinkingConfig(thinking_level=level),
        max_output_tokens=16384,
    )
    try:
        response = client.models.generate_content(model=model, contents=contents, config=gen_cfg)
    except Exception as e:
        if "thinking" not in str(e).lower():
            raise
        # questo modello non accetta il livello scelto: riprova con le impostazioni di default
        response = client.models.generate_content(model=model, contents=contents)
    return response.text or "(risposta vuota)"


def call_openai_compatible(cfg, key, model, prompt, imgs, context):
    client = OpenAI(
        base_url=cfg["base_url"],
        api_key=key,
        timeout=cfg.get("timeout", REQUEST_TIMEOUT),
        max_retries=0,  # i tentativi li gestisce run_provider
    )
    text = build_prompt(prompt, context)

    if imgs:
        content = [{"type": "text", "text": text}]
        for im in imgs:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    else:
        content = text  # solo testo: stringa semplice, la più compatibile

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    answer = response.choices[0].message.content or "(risposta vuota)"
    used_model = getattr(response, "model", None)
    if used_model and used_model != model and not cfg.get("_plain"):
        answer += f"\n\n*Modello usato: `{used_model}`*"
    return answer


def try_model(cfg, key, model, prompt, imgs, chunks, ranked, limit):
    """Chiama un modello con i suoi tentativi:
    - errori temporanei (503/429): riprova con una breve attesa (se cfg lo prevede);
    - richiesta troppo grande (413): dimezza il testo dei PDF e riprova.
    Restituisce un dizionario con l'esito."""
    fn = call_gemini if cfg["type"] == "gemini" else call_openai_compatible
    t0 = time.time()
    retries = shrinks = 0
    while True:
        context, sources = select_context(chunks, ranked, limit)
        try:
            text = fn(cfg, key, model, prompt, imgs, context)
            return {"ok": True, "text": text, "sources": sources, "chars": len(context),
                    "secs": time.time() - t0, "limit": limit}
        except Exception as e:
            low = str(e).lower()
            msg = str(e).replace("\n", " ")[:160]
            if chunks and shrinks < 3 and limit > 3000 and any(t in low for t in TOO_LARGE):
                shrinks += 1
                limit //= 2
                continue
            day_quota = any(t in low for t in DAY_MARKERS)  # inutile aspettare: fallisci subito
            if retries < cfg.get("retries", 2) and any(t in low for t in RETRYABLE) and not day_quota:
                retries += 1
                wait = 2 * retries  # 2s poi 4s
                headers = getattr(getattr(e, "response", None), "headers", None) or {}
                try:
                    wait = min(float(headers.get("retry-after")), 10)
                except (TypeError, ValueError):
                    pass
                time.sleep(wait)
                continue
            return {"ok": False, "error": f"`{model}`: {msg}", "secs": time.time() - t0, "limit": limit}


def run_models(cfg, key, models, prompt, imgs, chunks, ranked, start):
    """Prova i modelli in sequenza; se cfg ha 'hedge_after' li fa partire 'in gara':
    dopo quei secondi senza risposta (o appena uno fallisce) parte il modello successivo
    e vince la prima risposta valida.
    Restituisce (modello vincente o None, esito vincente, tentativi [(modello, esito)])."""
    hedge = cfg.get("hedge_after")
    attempts = []

    if not hedge or len(models) == 1:
        limit = cfg["max_chars"]
        for model in models:
            if time.time() - start > PROVIDER_BUDGET:
                attempts.append(("", {"ok": False, "secs": 0,
                                      "error": "⏱️ tempo massimo raggiunto, modelli successivi non provati"}))
                break
            r = try_model(cfg, key, model, prompt, imgs, chunks, ranked, limit)
            limit = r["limit"]
            attempts.append((model, r))
            if r["ok"]:
                return model, r, attempts
        return None, None, attempts

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=len(models))
    pending, nxt, winner = {}, 0, None

    def launch():
        nonlocal nxt
        model = models[nxt]
        nxt += 1
        pending[ex.submit(try_model, cfg, key, model, prompt, imgs, chunks, ranked, cfg["max_chars"])] = (
            model, time.time())

    launch()
    try:
        while pending and winner is None:
            done, _ = concurrent.futures.wait(
                list(pending),
                timeout=hedge if nxt < len(models) else None,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:  # il modello in corso è lento: avvia anche il successivo
                launch()
                continue
            for f in done:
                model, _t = pending.pop(f)
                r = f.result()
                attempts.append((model, r))
                if r["ok"] and winner is None:
                    winner = (model, r)
            if winner is None and nxt < len(models):
                launch()  # uno è fallito: avvia subito il successivo
    finally:
        ex.shutdown(wait=False)  # le richieste lente restano in sospeso ma non ci bloccano
    for f, (model, t) in pending.items():
        attempts.append((model, {"ok": False, "cancelled": True, "secs": time.time() - t,
                                 "error": f"`{model}`: nessuna risposta (scartato)"}))
    return (winner[0], winner[1], attempts) if winner else (None, None, attempts)


def fmt_num(n):
    return f"{n:,}".replace(",", ".")


def fmt_attempts(attempts):
    """'gemini-3-flash-preview: scartato dopo 12s · gemini-3.8-flash: ok in 3.1s'"""
    parts = []
    for model, a in attempts:
        if not model:
            continue
        if a["ok"]:
            parts.append(f"{model}: ok in {a['secs']:.1f}s")
        elif a.get("cancelled"):
            parts.append(f"{model}: nessuna risposta, scartato dopo {a['secs']:.0f}s")
        else:
            err = a.get("error", "").lower()
            reason = ("503" if ("503" in err or "unavailable" in err) else
                      "429" if "429" in err else
                      "timeout" if ("timed out" in err or "timeout" in err) else
                      "404" if "404" in err else "errore")
            parts.append(f"{model}: {reason} dopo {a['secs']:.1f}s")
    return " · ".join(parts)


def run_provider(cfg, key, prompt, imgs, chunks, ranked, is_answer=True):
    """Interroga un provider (provando i suoi modelli di riserva) e restituisce (testo, secondi).
    is_answer=False (es. trascrizione di un'immagine): niente formato finale né note."""
    if is_answer:
        prompt = FINAL_HEAD + prompt + FINAL_INSTRUCTION
    start = time.time()
    models = cfg["model"] if isinstance(cfg["model"], list) else [cfg["model"]]
    winner, r, attempts = run_models(cfg, key, models, prompt, imgs, chunks, ranked, start)
    if winner is None:
        errors = [a["error"] for _, a in attempts if a.get("error")]
        detail = "\n".join(f"- {e}" for e in errors) or "nessun modello configurato"
        return f"⚠️ Tutti i modelli hanno fallito:\n{detail}", time.time() - start
    text = r["text"]
    if is_answer:
        if winner != models[0]:
            text += f"\n\n*Modello di riserva usato: `{winner}`*"
        if r["sources"]:
            text += f"\n\n*📄 Fonti usate — {fmt_sources(r['sources'])} · {fmt_num(r['chars'])} caratteri*"
        if len(attempts) > 1:
            text += f"\n\n*⚙️ Tentativi — {fmt_attempts(attempts)}*"
    return text, time.time() - start


def synth_candidates(pool, has_images, quiz=False):
    """Chi può fare la sintesi, in ordine: Gemini, altri provider con visione e (se non ci
    sono immagini) gli altri. Se ci sono immagini servono provider con visione."""
    vision = [p for p in pool if p[0]["vision"]]
    gem = [p for p in vision if p[0]["type"] == "gemini"]
    rest = [p for p in vision if p[0]["type"] != "gemini"]
    others = [] if has_images else [p for p in pool if not p[0]["vision"]]
    def limit(c):
        cap = SYNTH_QUIZ_CHARS if (quiz and c["type"] == "gemini") else SYNTH_MAX_CHARS
        return min(c["max_chars"], cap)

    def tune(c):
        c = dict(c, max_chars=limit(c))
        if c["type"] == "gemini" and c.get("thinking", "low") == "low":
            c["thinking"] = "medium"  # per verificare i calcoli serve più cura
        return c

    return [(tune(c), k) for c, k in gem + rest + others]


def build_synthesis_prompt(question, valid, quiz=False):
    """Domanda + risposte complete di tutte le AI (senza note tecniche) + compito."""
    parts = []
    for name, res in valid.items():
        text, _ = split_footer(res)
        parts.append(f"--- {short(name)} ---\n{text[:SYNTH_MAX_ANSWER]}")
    return (
        SYNTH_HEAD
        + f"DOMANDA DELL'UTENTE:\n{question}\n\n"
        + "=== RISPOSTE DELLE IA (con i loro ragionamenti) ===\n"
        + "\n\n".join(parts)
        + "\n=== FINE RISPOSTE ===\n\n"
        + SYNTH_TASK
        + (QUIZ_SYNTH_NOTE if quiz else "")
    )


def run_synthesis(synth_prompt, imgs, chunks, ranked, candidates):
    """Prova al massimo due candidati. Restituisce {'text','secs','by'} oppure None."""
    for cfg, key in candidates[:2]:
        text, secs = run_provider(cfg, key, synth_prompt, imgs, chunks, ranked)
        if not text.startswith("⚠️"):
            return {"text": text, "secs": secs, "by": cfg["name"]}
    return None


def transcribe_image(img, vision_providers):
    """Fa leggere un'immagine al primo provider con visione che risponde.
    Restituisce (testo, None) oppure (None, errore)."""
    errors = []
    for cfg, key in vision_providers:
        plain_cfg = dict(cfg, _plain=True, thinking="low")  # senza note; leggere non richiede ragionare
        text, _ = run_provider(plain_cfg, key, TRANSCRIBE_PROMPT, [img], [], [], is_answer=False)
        if not text.startswith("⚠️"):
            return text, None
        errors.append(f"{cfg['name']}: {text}".replace("\n", " ")[:300])
    return None, " | ".join(errors) or "nessun provider con visione disponibile"


def transcribe_images(images, vision_providers, cache):
    """Trascrive tutte le immagini. Quelle già lette in precedenza (es. file fissi)
    vengono riprese dalla cache, senza nuove chiamate. Restituisce (testo, None) o (None, errore)."""
    todo = [im for im in images if im["id"] not in cache]
    errs = {}
    if todo:
        def work(im):
            return im["id"], transcribe_image(im["img"], vision_providers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(todo))) as ex:
            for iid, (text, err) in ex.map(work, todo):
                if text:
                    cache[iid] = text
                else:
                    errs[iid] = err
    if not any(im["id"] in cache for im in images):
        return None, " | ".join(errs.values())[:300] or "nessuna immagine leggibile"
    parts = []
    for n, im in enumerate(images, start=1):
        head = f"--- Immagine {n} ({im['name']}) ---"
        if im["id"] in cache:
            parts.append(f"{head}\n{cache[im['id']]}")
        else:
            parts.append(f"{head}\n[non leggibile: {errs.get(im['id'], 'errore')}]")
    return "\n\n".join(parts), None


def run_with_transcript(cfg, key, prompt, transcript_future, chunks, ranked):
    """Per i provider senza visione: aspetta la trascrizione e la aggiunge alla domanda."""
    start = time.time()
    transcript, err = transcript_future.result()
    if transcript is None:
        return f"⚠️ Non sono riuscito a leggere le immagini: {err}", time.time() - start
    full_prompt = (
        "L'utente ha allegato una o più immagini. Il loro contenuto è già stato estratto "
        "automaticamente (può contenere piccoli errori) ed è riportato qui sotto: "
        "usalo come se vedessi tu stesso le immagini.\n\n"
        "=== CONTENUTO DELLE IMMAGINI ===\n"
        f"{transcript}\n"
        "=== FINE CONTENUTO ===\n\n"
        f"Richiesta dell'utente: {prompt}\n\n"
        "Rispondi basandoti sul contenuto delle immagini qui sopra. "
        "Non dire di non poter vedere le immagini e non chiedere di ricopiarle."
    )
    result, _ = run_provider(cfg, key, full_prompt, [], chunks, ranked)
    return result, time.time() - start


# -----------------------------------------------------------------------------
# VISUALIZZAZIONE RISULTATI
# -----------------------------------------------------------------------------
def split_final(text):
    """Separa la 'RISPOSTA FINALE: ...' dal resto.
    Restituisce (risposta o None, testo senza quella riga)."""
    matches = list(FINAL_RE.finditer(text))
    if not matches:
        return None, text
    m = matches[-1]
    answer = m.group("ans").strip().strip("*_`").strip()
    letter = re.fullmatch(r"\(?([A-Za-z])\)?[.)]?", answer)  # '(b)', 'B)', 'b.' -> 'b'
    if letter:
        answer = letter.group(1)
    elif answer.endswith(".") and not answer.endswith(".."):
        answer = answer[:-1].rstrip()  # '4.' -> '4' ('3.5' resta invariato)
    if len(answer) > 200:  # una "risposta" così lunga è un ragionamento: la accorcio
        answer = answer[:197].rstrip() + "…"
    body = (text[: m.start()] + text[m.end():]).strip()
    return (answer or None), body


def split_footer(text):
    """Separa le note finali (modello usato, fonti, tentativi) dal testo.
    Restituisce (testo, [note])."""
    paras = re.split(r"\n\s*\n", text.strip())
    footer = []
    while paras:
        last = paras[-1].strip()
        if last.startswith("*") and last.endswith("*") and any(m in last for m in FOOTER_MARKERS):
            footer.insert(0, paras.pop().strip().strip("*").strip())
        else:
            break
    return "\n\n".join(paras).strip(), footer


def guess_answer(body):
    """Se il modello non ha scritto 'RISPOSTA FINALE', prova a ricavarla dal testo:
    una frase tipo 'la risposta è X', altrimenti l'ultima frase."""
    lines = [re.sub(r"[*_`#>]", "", l).strip() for l in body.splitlines() if l.strip()]
    if not lines:
        return None
    tail = " ".join(lines[-3:])
    hits = re.findall(
        r"(?:risposta(?:\s+(?:finale|corretta))?|soluzione|risultato)\s*(?:è|e'|:)\s*([^.!?\n]{1,100})",
        tail, flags=re.IGNORECASE)
    if hits:
        return hits[-1].strip(" :;,")
    last = lines[-1]
    if len(last) > 160:
        last = re.split(r"(?<=[.!?])\s+", last)[-1]
    return last[:160].strip() or None


def is_quota_error(text):
    low = text.lower()
    return any(m in low for m in QUOTA_MARKERS)


def norm_answer(answer):
    """Forma confrontabile: 'B)', '(b)' e 'b.' diventano 'b'; '60 N' e '60N' coincidono."""
    a = answer.lower().strip(" .)(\"'`*_")
    return re.sub(r"\s+", "", a)


def show_pending(slot):
    slot.info("⏳ In corso...")


def show_result(slot, result, seconds, compact=False):
    """Riquadro con la risposta finale, poi il ragionamento, poi le note e il tempo.
    Con 'compact' il ragionamento sta sempre in un riquadro chiuso, anche se il modello
    non ha scritto 'RISPOSTA FINALE' (in quel caso la risposta viene ricavata dal testo)."""
    answer, body = split_final(result)
    body, footer = split_footer(body)
    with slot.container():
        if result.startswith("⚠️"):
            detail = result.replace("⚠️", "", 1).strip()
            if is_quota_error(detail):
                st.warning("Quota gratuita esaurita o troppe richieste. L'AI resta attiva: "
                           "verrà riprovata alla prossima domanda.")
            else:
                st.warning("Questa AI non è riuscita a rispondere.")
            with st.expander("Dettagli dell'errore"):
                st.markdown(detail)
        else:
            guessed = False
            if compact and not answer:
                answer, guessed = guess_answer(body), True
            if answer:
                if guessed:
                    st.info(f"**{answer}**", icon="❔")
                    st.caption("Il modello non ha indicato la risposta finale: estratta dal testo.")
                else:
                    st.success(f"**{answer}**", icon="✅")
            if compact and answer:
                with st.expander("Ragionamento"):
                    st.markdown(body)
            else:
                st.markdown(body)
            if footer:
                st.caption(" · ".join(footer))
        st.caption(f"⏱️ {seconds:.1f}s")


def show_consensus(slot, answers, total, finished, extra=""):
    """Barra riassuntiva sopra le colonne: le AI concordano?
    answers = {nome AI: risposta finale}; finished = quante AI hanno già terminato."""
    pending = total - finished
    if not answers:
        if pending:
            slot.info(f"⏳ In attesa delle risposte… (0/{total})")
        else:
            slot.warning("Nessuna AI ha fornito una risposta finale." + extra)
        return

    groups = {}  # risposta normalizzata -> [testo, [AI]]
    for name, ans in answers.items():
        groups.setdefault(norm_answer(ans), [ans, []])[1].append(short(name))
    ranked = sorted(groups.values(), key=lambda g: len(g[1]), reverse=True)
    top_ans, top_names = ranked[0]
    n = len(answers)
    note = (f" · {total - n} senza risposta finale" if (not pending and n < total) else "") + extra

    if pending:
        slot.info(f"⏳ {finished}/{total} risposte · finora: **{top_ans}** ({len(top_names)})")
    elif n == 1:
        slot.info(f"Ha risposto una sola AI ({top_names[0]}): **{top_ans}**{note}")
    elif len(top_names) == n:
        slot.success(f"✅ Tutte le AI concordano: **{top_ans}**{note}")
    elif len(top_names) > n / 2:
        others = " · ".join(f"{', '.join(g[1])}: **{g[0]}**" for g in ranked[1:])
        slot.success(f"🟢 Maggioranza: **{top_ans}** ({len(top_names)} su {n}) — diversa: {others}{note}")
    else:
        detail = " · ".join(f"**{g[0]}** ← {', '.join(g[1])}" for g in ranked)
        slot.warning(f"⚠️ Risposte discordanti: {detail}{note}")


def show_synthesis(slot, synth):
    """Report finale di sintesi (al posto della vecchia barra 'le AI concordano')."""
    answer, body = split_final(synth["text"])
    body, footer = split_footer(body)
    with slot.container():
        with st.container(border=True):
            st.markdown("##### 🧠 Sintesi finale")
            if answer:
                st.success(f"**{answer}**", icon="✅")
            st.markdown(body)
            st.caption(f"⏱️ {synth['secs']:.1f}s · sintesi di {short(synth['by'])}")


def show_transcript(slot, transcript):
    if not transcript:
        return
    names = ", ".join(short(n) for n in transcript["names"])
    with slot.container():
        with st.expander(f"📝 Testo letto dalle immagini (passato a: {names})"):
            st.markdown(transcript["text"] if transcript["text"] else f"⚠️ {transcript['err']}")


def build_layout(question, attachments, names):
    """Barra di consenso, trascrizione e una card per ogni AI. Restituisce gli slot da riempire."""
    st.caption(f"💬 {question[:200]}")
    if attachments:
        st.caption("📎 " + " · ".join(attachments))
    consensus_slot = st.empty()
    transcript_slot = st.empty()
    slots = {}
    for col, name in zip(st.columns(len(names)), names):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {name}")
                slots[name] = st.empty()
    return consensus_slot, transcript_slot, slots


def render_static(run, compact):
    """Ridisegna l'ultima esecuzione (dopo un rerun, es. cambio impostazioni)."""
    names = run["names"]
    consensus_slot, transcript_slot, slots = build_layout(run["question"], run["attachments"], names)
    answers = {}
    for name in names:
        result, seconds = run["results"][name]
        show_result(slots[name], result, seconds, compact)
        ans, _ = split_final(result)
        if ans:
            answers[name] = ans
    if run.get("synthesis"):
        show_synthesis(consensus_slot, run["synthesis"])
    else:
        show_consensus(consensus_slot, answers, len(names), len(names), run.get("synth_note", ""))
    show_transcript(transcript_slot, run["transcript"])


def execute_run(prompt_text, images, chunks, attachments, active, compact, pool, use_synth, quiz_mode=False):
    """Lancia tutte le AI in parallelo e riempie le colonne man mano che rispondono."""
    prompt = prompt_text.strip() or "Risolvi il quesito presente nell'allegato."
    if quiz_mode:
        prompt = QUIZ_PROMPT + prompt
    names = [c["name"] for c, _ in active]

    vision_active = [(c, k) for c, k in active if c["vision"]]
    text_only_names = [c["name"] for c, _ in active if not c["vision"]]
    needs_transcript = bool(images) and bool(text_only_names)
    if images and not vision_active:
        st.error("Per le immagini serve almeno un provider con visione (Gemini o OpenRouter).")
        return None

    # Classifica degli estratti dei PDF rispetto alla domanda (una volta sola)
    ranked = rank_chunks(chunks, prompt_text) if chunks else []
    direct_imgs = [im["img"] for im in images]

    question = prompt_text.strip() or "(solo allegati)"
    consensus_slot, transcript_slot, slots = build_layout(question, attachments, names)
    show_consensus(consensus_slot, {}, len(names), 0)
    for slot in slots.values():
        show_pending(slot)

    results, answers, transcript = {}, {}, None

    # +1 worker per la trascrizione delle immagini (evita blocchi in attesa)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active) + 1) as executor:
        futures = {}
        transcript_future = None
        if needs_transcript:
            transcript_future = executor.submit(
                transcribe_images, images, vision_active, st.session_state.transcripts
            )
            futures[transcript_future] = None  # None = trascrizione
        for cfg, key in active:
            if needs_transcript and not cfg["vision"]:
                f = executor.submit(run_with_transcript, cfg, key, prompt, transcript_future, chunks, ranked)
            else:
                f = executor.submit(run_provider, cfg, key, prompt, direct_imgs, chunks, ranked)
            futures[f] = cfg

        for future in concurrent.futures.as_completed(futures):
            cfg = futures[future]
            if cfg is None:
                text, err = future.result()
                transcript = {"text": text, "err": err, "names": text_only_names}
                show_transcript(transcript_slot, transcript)
                continue
            result, seconds = future.result()
            results[cfg["name"]] = (result, seconds)
            ans, _ = split_final(result)
            if ans:
                answers[cfg["name"]] = ans
            show_result(slots[cfg["name"]], result, seconds, compact)
            if len(results) < len(names):  # l'esito finale lo scrive il blocco dopo il ciclo
                show_consensus(consensus_slot, answers, len(names), len(results))

    # Le AI che hanno fallito in questa domanda non vengono usate per la sintesi
    # (restano comunque attive e visibili, e saranno riprovate alla prossima domanda)
    failed = {n for n, (r, _) in results.items() if r.startswith("⚠️")}

    # Sintesi finale: tutte le risposte e i ragionamenti vengono dati in pasto a Gemini
    synthesis, synth_note = None, ""
    valid = {n: r for n, (r, _) in results.items() if not r.startswith("⚠️")}
    if use_synth and len(valid) >= 2:
        candidates = [c for c in synth_candidates(pool, bool(images), quiz_mode) if c[0]["name"] not in failed]
        if candidates:
            consensus_slot.info(
                f"🧠 Sintesi in corso: {short(candidates[0][0]['name'])} confronta le risposte "
                "e i ragionamenti di tutte le AI…"
            )
            synth_prompt = build_synthesis_prompt(question if prompt_text.strip() else
                                                  "(vedi gli allegati)", valid, quiz_mode)
            synthesis = run_synthesis(synth_prompt, direct_imgs, chunks, ranked, candidates)
            if synthesis:
                show_synthesis(consensus_slot, synthesis)
            else:
                synth_note = " · sintesi non riuscita"
    if not synthesis:
        show_consensus(consensus_slot, answers, len(names), len(names), synth_note)

    return {
        "question": question,
        "attachments": attachments,
        "names": names,
        "results": results,
        "transcript": transcript,
        "synthesis": synthesis,
        "synth_note": synth_note,
    }


def add_history(run):
    answers = {n: split_final(res)[0] for n, (res, _) in run["results"].items()}
    if run.get("synthesis"):
        answers = {"🧠 Sintesi": split_final(run["synthesis"]["text"])[0], **answers}
    st.session_state.history.insert(
        0, {"q": run["question"], "answers": answers, "time": datetime.now().strftime("%H:%M")}
    )
    del st.session_state.history[HISTORY_SIZE:]


# -----------------------------------------------------------------------------
# STATO
# -----------------------------------------------------------------------------
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "last_run" not in st.session_state:
    st.session_state.last_run = None
if "lib_n" not in st.session_state:
    st.session_state.lib_n = 0  # cambia dopo ogni caricamento per svuotare il campo
if "transcripts" not in st.session_state:
    st.session_state.transcripts = {}  # id immagine -> testo letto (evita riletture)


def do_clear():
    st.session_state.reset_trigger += 1  # nuove chiavi = campi vuoti
    st.session_state.last_run = None


def clear_history():
    st.session_state.history = []



key_suffix = str(st.session_state.reset_trigger)

# -----------------------------------------------------------------------------
# BARRA LATERALE: scelta delle AI, file fissi e opzioni
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Impostazioni")
    st.caption("AI da interrogare")
    enabled = {}
    for cfg in PROVIDERS:
        if get_key(cfg["secret"]):
            enabled[cfg["name"]] = st.checkbox(cfg["name"], value=True, key="use_" + cfg["name"])
    if not enabled:
        st.warning("Nessuna chiave API trovata in secrets.toml.")
    gemini_level = "low"
    if any(c["type"] == "gemini" and get_key(c["secret"]) for c in PROVIDERS):
        gemini_level = st.selectbox(
            "Ragionamento di Gemini",
            list(GEMINI_LEVELS),
            index=0,
            format_func=lambda k: GEMINI_LEVELS[k][0],
            help="Veloce risponde in pochi secondi; Approfondito ragiona di più sui quesiti "
                 "difficili ma può metterci anche decine di secondi.",
        )
    quiz_mode = st.checkbox(
        "🎯 Modalità quiz (precisione)",
        value=True,
        help="Per quiz sul regolamento: istruzioni da esaminatore (valuta ogni opzione, cita la regola), "
             "a Gemini va tutto il regolamento invece dei soli estratti e il ragionamento è almeno "
             "«Bilanciato». Più preciso ma più lento: toglila per domande veloci.",
    )
    use_synth = st.checkbox(
        "🧠 Sintesi finale",
        value=True,
        help="Quando tutte le AI hanno risposto, Gemini legge le risposte e i ragionamenti di "
             "tutte (anche il proprio) e scrive un report con la risposta su cui si accorda. "
             "Aggiunge qualche secondo.",
    )
    compact = st.checkbox(
        "Ragionamento ripiegato",
        value=False,
        help="Mostra solo il riquadro verde con la risposta; il ragionamento si apre con un clic.",
    )

    st.divider()
    st.header("📚 File fissi")
    st.caption(
        f"Caricali una volta: restano salvati nella cartella `{LIB_DIR.name}` e vengono "
        "inviati a ogni domanda (anche dopo aver riavviato l'app)."
    )
    new_files = st.file_uploader(
        "Aggiungi file fissi",
        type=FILE_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"lib_up_{st.session_state.lib_n}",
    )
    if new_files:
        lib_save(new_files)
        st.session_state.lib_n += 1
        rerun()

    use_lib = {}
    for path in lib_files():
        c_name, c_del = st.columns([5, 1])
        with c_name:
            use_lib[path.name] = st.checkbox(path.name, value=True, key="libuse_" + path.name)
        with c_del:
            st.button("🗑️", key="libdel_" + path.name, on_click=lib_delete, args=(path.name,),
                      help="Elimina dalla libreria")
    if use_lib:
        st.caption(f"{sum(use_lib.values())} di {len(use_lib)} file fissi attivi")
    else:
        st.caption("Nessun file fisso.")

# -----------------------------------------------------------------------------
# INPUT
# - La casella di testo sta in un form: Ctrl+Invio = Confronta, e la pagina non si
#   ricarica a ogni modifica. È disegnata per prima, così compare sempre.
# - Il caricamento file sta FUORI dal form (così funziona subito) ma viene mostrato
#   sopra la casella di testo grazie al contenitore "attach_box".
# -----------------------------------------------------------------------------
attach_box = st.container()

with st.form("mas_form"):
    prompt_text = st.text_area(
        "Quesito",
        height=110,
        placeholder="Scrivi qui il tuo quesito o incolla il testo...  (Ctrl+Invio per confrontare)",
        label_visibility="collapsed",
        key=f"prompt_{key_suffix}",
    )
    col_a, col_b, _ = st.columns([1, 1, 6])
    with col_a:
        submit_clicked = st.form_submit_button("🚀 Confronta", type="primary")
    with col_b:
        st.form_submit_button("🗑️ Cancella", on_click=do_clear)

pasted_image = None
with attach_box:
    col_up, col_paste = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Allegato per questa domanda:",
            type=FILE_TYPES,
            label_visibility="collapsed",
            key=f"file_{key_suffix}",
        )
    with col_paste:
        if paste_image_button is not None:
            # Win+Shift+S, poi clic qui (il browser può chiedere il permesso agli appunti)
            try:
                paste_result = paste_image_button("📋 Incolla screenshot", key=f"paste_{key_suffix}")
                if paste_result.image_data is not None:
                    pasted_image = paste_result.image_data
            except Exception as e:  # un problema del pulsante non deve bloccare l'app
                st.caption(f"Pulsante incolla non disponibile: {e}")
        else:
            st.caption("Per incollare screenshot con un clic: `pip install streamlit-paste-button` e riavvia.")

# Allegati di QUESTA domanda
query_images, query_chunks, query_names = [], [], []

if uploaded_file is not None:
    ext = uploaded_file.name.split(".")[-1].lower()
    data = uploaded_file.getvalue()
    if ext in ("png", "jpg", "jpeg"):
        query_images.append(image_entry(data, uploaded_file.name))
        query_names.append(uploaded_file.name)
        st.info(f"🖼️ Immagine allegata: **{uploaded_file.name}**")
    elif ext == "pdf":
        try:
            query_chunks += extract_pdf_chunks(data, uploaded_file.name)
            query_names.append(uploaded_file.name)
            st.info(f"📄 Documento PDF allegato: **{uploaded_file.name}**")
            if not query_chunks:
                st.warning("Nessun testo estraibile dal PDF (forse è una scansione).")
        except Exception as e:
            st.error(f"Errore nella lettura del PDF (hai installato pypdf?): {e}")

if pasted_image is not None:
    buf = io.BytesIO()
    pasted_image.save(buf, format="PNG")
    query_images.append(image_entry(buf.getvalue(), "screenshot"))
    query_names.append("screenshot incollato")
    st.info("🖼️ Screenshot incollato dagli appunti")

# -----------------------------------------------------------------------------
# ESECUZIONE
# -----------------------------------------------------------------------------
if submit_clicked:
    if not prompt_text.strip() and not query_names:
        st.warning("Inserisci del testo o allega un file prima di inviare.")
    else:
        pool = []  # tutti i provider con chiave (la sintesi può usarli anche se disattivati)
        for cfg in PROVIDERS:
            key = get_key(cfg["secret"])
            if not key:
                continue
            if cfg["type"] == "gemini":
                level = "medium" if (quiz_mode and gemini_level == "low") else gemini_level
                cfg = dict(cfg, thinking=level)
                if quiz_mode:  # documento intero: serve più tempo e niente richieste "in gara" raddoppiate
                    cfg.update(max_chars=QUIZ_GEMINI_CHARS, hedge_after=25, timeout_bonus=30)
            pool.append((cfg, key))
        active = [(c, k) for c, k in pool if enabled.get(c["name"], False)]
        if not active:
            st.error("Nessuna AI selezionata: attivane almeno una nella barra laterale "
                     "(e controlla le chiavi in secrets.toml).")
        else:
            with st.spinner("Leggo i file..."):
                lib_chunks, lib_images, lib_names = load_library(use_lib)
            images = lib_images + query_images
            chunks = lib_chunks + query_chunks
            attachments = [f"{n} (fisso)" for n in lib_names] + query_names
            st.markdown("---")
            run = execute_run(prompt_text, images, chunks, attachments, active, compact, pool, use_synth, quiz_mode)
            if run:
                st.session_state.last_run = run
                add_history(run)
elif st.session_state.last_run:
    st.markdown("---")
    render_static(st.session_state.last_run, compact)

# -----------------------------------------------------------------------------
# CRONOLOGIA (barra laterale)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("🕘 Cronologia")
    if not st.session_state.history:
        st.caption("Ancora nessuna domanda.")
    else:
        for h in st.session_state.history[:10]:
            with st.expander(f"{h['time']} · {h['q'][:40]}"):
                for name, ans in h["answers"].items():
                    st.markdown(f"{name}: **{ans or '—'}**")
        st.button("🧹 Svuota cronologia", on_click=clear_history)
