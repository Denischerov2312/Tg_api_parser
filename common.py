"""
Общий код для parser.py и dedup_llm.py: настройки LLM, хранилище событий,
правила поиска дублей, выгрузка в CSV, логи и статистика запусков.
"""
import csv
import json
import logging
import os
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import AsyncOpenAI

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.proxyapi.ru/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4.1-flash")
TITLE_SIMILARITY = float(os.getenv("TITLE_SIMILARITY", "0.85"))  # порог похожести названий для правила C

EVENTS_FILE = BASE / "events.json"   # все найденные события (основное хранилище)
OUTPUT_FILE = BASE / "events.csv"    # актуальные события для Excel, пересобирается каждый запуск
LOGS_DIR = BASE / "logs"

# Поля события, которые извлекает LLM
EVENT_FIELDS = [
    "start_date", "end_date", "registration_deadline", "kind", "title", "summary",
    "format", "city", "participation", "audience", "prize", "registration_url",
]
CSV_FIELDS = ["id"] + EVENT_FIELDS + ["sources_count", "sources"]

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Слова, которые не помогают отличить одно событие от другого
GENERIC_WORDS_RE = re.compile(
    r"\b(20\d\d|хакатон|hackathon|чемпионат|конкурс|олимпиада|молодежный|"
    r"международный|всероссийский)\b"
)


# ---------- логи и статистика ----------

def setup_logging(name: str) -> logging.Logger:
    """Консоль: короткие сообщения. Файл logs/<name>.log: всё подробно, с датой."""
    LOGS_DIR.mkdir(exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    file = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(console)
    log.addHandler(file)
    return log


def append_stats(path: Path, fields: list, row: dict):
    """Дописывает строку статистики запуска. Если набор колонок поменялся,
    старый файл переименовывается, и начинается новый."""
    LOGS_DIR.mkdir(exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8-sig") as f:
            header = f.readline().strip().split(";")
        if header != fields:
            path.replace(path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"))
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";", extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


class LLMUsage:
    """Счётчики обращений к модели за запуск."""

    def __init__(self):
        self.requests = 0       # все HTTP-запросы, включая повторные
        self.bad_json = 0       # ответы, которые не удалось разобрать как JSON
        self.prompt_tokens = 0
        self.completion_tokens = 0


def make_llm() -> AsyncOpenAI:
    if not LLM_API_KEY:
        raise SystemExit("Заполните DEEPSEEK_API_KEY в файле .env")
    return AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=180)


async def llm_json(llm: AsyncOpenAI, system: str, user: str, usage: LLMUsage, attempts: int = 2):
    """Запрос к модели в JSON-режиме. Возвращает dict или None, если ответ так и не разобрался.
    Ошибки API (сеть, баланс, лимиты) пробрасываются наверх."""
    for _ in range(attempts):
        resp = await llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_tokens=8000,  # с запасом: модель тратит часть токенов на рассуждение
        )
        usage.requests += 1
        if resp.usage:
            usage.prompt_tokens += resp.usage.prompt_tokens or 0
            usage.completion_tokens += resp.usage.completion_tokens or 0
        try:
            data = json.loads(resp.choices[0].message.content or "")
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        usage.bad_json += 1
    return None


def pct(part: int, total: int) -> float:
    return round(100 * part / total, 1) if total else 0.0


# ---------- нормализация полей ----------

def clean_date(value) -> str:
    if isinstance(value, str) and DATE_RE.match(value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value
        except ValueError:
            pass
    return ""


def clean_str(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def is_actual(event: dict, today: str) -> bool:
    """Событие ещё не прошло: хотя бы одна известная дата не раньше сегодняшней.
    Если дат нет совсем, оставляем, чтобы не потерять анонс."""
    known = [d for d in (event["end_date"] or event["start_date"], event["registration_deadline"]) if d]
    return not known or max(known) >= today


# ---------- хранилище событий ----------

def load_events() -> list:
    if EVENTS_FILE.exists():
        return json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
    return []


def save_events(events: list):
    EVENTS_FILE.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")


def next_id(events: list) -> int:
    return max((e["id"] for e in events), default=0) + 1


def known_post_links(events: list) -> set:
    return {s["post_link"] for e in events for s in e["sources"]}


def merge_into(target: dict, other: dict):
    """Переносит в target источники и непустые поля из other (заполняет только пустые)."""
    links = {s["post_link"] for s in target["sources"]}
    for s in other["sources"]:
        if s["post_link"] not in links:
            target["sources"].append(s)
    for f in EVENT_FIELDS:
        if not target.get(f) and other.get(f):
            target[f] = other[f]
    target["updated_at"] = datetime.now().isoformat(timespec="seconds")


# ---------- правила дублей ----------

def norm_url(url: str) -> str:
    """Ссылка без протокола, www, параметров и якоря. Голый домен и ссылки на Telegram
    не считаются признаком события: у одного организатора они одинаковые для разных событий."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/").lower()
    if not host or not path or host in ("t.me", "telegram.me"):
        return ""
    return host + path


def norm_title(title: str) -> str:
    t = title.lower().replace("ё", "е")
    t = GENERIC_WORDS_RE.sub(" ", t)
    t = re.sub(r"[^\w ]", "", t)
    return " ".join(t.split())


def title_similarity(a: str, b: str) -> float:
    a, b = norm_title(a), norm_title(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def find_duplicate(events: list, event: dict):
    """Ищет уже известное событие по правилам B и C. Возвращает (событие, "url"/"title") или (None, None)."""
    url = norm_url(event["registration_url"])
    if url:
        for e in events:
            if norm_url(e["registration_url"]) == url:
                return e, "url"
    if event["start_date"]:
        for e in events:
            if e["start_date"] == event["start_date"] and \
                    title_similarity(e["title"], event["title"]) >= TITLE_SIMILARITY:
                return e, "title"
    return None, None


# ---------- CSV для Excel ----------

def export_csv(events: list, log: logging.Logger) -> int:
    """Пересобирает events.csv из актуальных событий. Возвращает число строк."""
    today = date.today().isoformat()
    actual = [e for e in events if is_actual(e, today)]
    actual.sort(key=lambda e: e["start_date"] or e["registration_deadline"] or "9999")
    try:
        # utf-8-sig, чтобы Excel корректно открыл кириллицу
        with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=";", extrasaction="ignore")
            w.writeheader()
            for e in actual:
                w.writerow({
                    **e,
                    "sources_count": len(e["sources"]),
                    "sources": " ".join(s["post_link"] for s in e["sources"]),
                })
    except PermissionError:
        log.warning(f"Не удалось записать {OUTPUT_FILE.name}: закройте его в Excel и запустите снова. "
                    f"Данные сохранены в {EVENTS_FILE.name}.")
    return len(actual)
