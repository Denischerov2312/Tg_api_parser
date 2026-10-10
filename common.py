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
LOGS_DIR = BASE / "logs"
OUTPUT_FILE = BASE / "events.csv"    # актуальные события для Excel, пересобирается каждый запуск

# Поля события, которые извлекает LLM
EVENT_FIELDS = [
    "start_date", "end_date", "registration_deadline", "kind", "title", "summary",
    "format", "city", "participation", "audience", "prize", "registration_url",
]
# Колонки events.csv: (ключ события, заголовок в Excel)
CSV_COLUMNS = [
    ("id", "№"), ("start_date", "Начало"), ("end_date", "Конец"),
    ("registration_deadline", "Регистрация до"), ("kind", "Тип"), ("title", "Название"),
    ("summary", "Что делать"), ("format", "Формат"), ("city", "Город"),
    ("participation", "Участие"), ("audience", "Для кого"), ("prize", "Призы"),
    ("registration_url", "Ссылка на регистрацию"), ("sources_count", "Упоминаний"),
    ("sources", "Посты"),
]

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Слова, которые не помогают отличить одно событие от другого
GENERIC_WORDS_RE = re.compile(
    r"\b(20\d\d|хакатон|hackathon|чемпионат|конкурс|олимпиада|молодежный|"
    r"международный|всероссийский)\b"
)


# ---------- логи и статистика ----------

RUNS_DIR = LOGS_DIR / "runs"   # отдельный подробный лог на каждый запуск
KEEP_RUN_LOGS = 100            # сколько последних логов каждого скрипта хранить
PARSER_STATS_FILE = LOGS_DIR / "parser_runs.csv"
DEDUP_STATS_FILE = LOGS_DIR / "dedup_runs.csv"

# Цена модели за 1 млн токенов (в валюте провайдера, у ProxyAPI — рубли). 0 = не считать стоимость
LLM_PRICE_INPUT = float(os.getenv("LLM_PRICE_INPUT", "0"))
LLM_PRICE_OUTPUT = float(os.getenv("LLM_PRICE_OUTPUT", "0"))

# Колонки статистики: (ключ в коде, заголовок в CSV)
PARSER_STATS_COLUMNS = [
    ("started_at", "Дата запуска"), ("status", "Статус"), ("error", "Ошибка"),
    ("duration_s", "Время, с"), ("model", "Модель"),
    ("channels", "Каналов"), ("channels_failed", "Каналов с ошибкой"),
    ("posts_read", "Прочитано постов"),
    ("dropped_short", "Отсеяно: короткие"), ("dropped_no_keywords", "Отсеяно: нет ключевых слов"),
    ("dropped_same_text", "Отсеяно: повтор текста"), ("dup_post_link", "Отсеяно: уже в базе"),
    ("sent_to_llm", "Отправлено в модель"), ("llm_failed", "Ошибок модели"),
    ("llm_error_pct", "Ошибок модели, %"), ("llm_requests", "Запросов к API"),
    ("llm_bad_json", "Невалидных ответов"), ("prompt_tokens", "Токенов на вход"),
    ("completion_tokens", "Токенов на выход"), ("cost", "Стоимость"),
    ("not_competition", "Не соревнование"), ("past", "Уже прошли"),
    ("dropped_not_online", "Не онлайн"), ("dup_url", "Дубли: ссылка на регистрацию"),
    ("dup_title", "Дубли: название и дата"), ("new_events", "Новых событий"),
    ("events_total", "Всего в базе"), ("events_actual", "Актуальных"), ("log_file", "Лог запуска"),
]
DEDUP_STATS_COLUMNS = [
    ("started_at", "Дата запуска"), ("status", "Статус"), ("error", "Ошибка"),
    ("duration_s", "Время, с"), ("model", "Модель"), ("dry_run", "Пробный запуск"),
    ("events_checked", "Проверено событий"), ("groups_returned", "Групп от модели"),
    ("groups_invalid", "Некорректных групп"), ("llm_error_pct", "Ошибок модели, %"),
    ("llm_requests", "Запросов к API"), ("llm_bad_json", "Невалидных ответов"),
    ("prompt_tokens", "Токенов на вход"), ("completion_tokens", "Токенов на выход"),
    ("cost", "Стоимость"), ("events_merged", "Склеено записей"),
    ("events_total_after", "Всего в базе после"), ("log_file", "Лог запуска"),
]


def setup_logging(name: str):
    """Консоль: главное. Файл logs/runs/<дата_время>_<name>.log: всё подробно, по строке на каждый пост.
    Возвращает (logger, путь к файлу лога)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for old in sorted(RUNS_DIR.glob(f"*_{name}.log"))[:-(KEEP_RUN_LOGS - 1)]:
        old.unlink()
    path = RUNS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{name}.log"

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    file = logging.FileHandler(path, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(console)
    log.addHandler(file)
    return log, path


def tag(label: str, text: str) -> str:
    """Строка лога с меткой в квадратных скобках, метки выровнены в колонку."""
    return f"{'[' + label + ']':<24} {text}"


def snippet(text: str, length: int = 80) -> str:
    """Начало текста поста в одну строку — чтобы в логе было видно, что это за пост."""
    one_line = " ".join(text.replace("**", "").replace("__", "").split())
    if not one_line:
        return "(без текста)"
    return one_line[:length] + ("…" if len(one_line) > length else "")


def llm_cost(prompt_tokens: int, completion_tokens: int):
    if not (LLM_PRICE_INPUT or LLM_PRICE_OUTPUT):
        return ""
    return round((prompt_tokens * LLM_PRICE_INPUT + completion_tokens * LLM_PRICE_OUTPUT) / 1_000_000, 4)


def append_stats(path: Path, columns: list, row: dict):
    """Дописывает строку статистики запуска (CSV с русскими заголовками для Excel).
    Если набор колонок поменялся, старый файл переименовывается, и начинается новый."""
    labels = [label for _, label in columns]
    LOGS_DIR.mkdir(exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8-sig") as f:
            header = f.readline().rstrip("\r\n").split(";")
        if header != labels:
            path.replace(path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"))
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        if write_header:
            w.writerow(labels)
        w.writerow([row.get(key, "") for key, _ in columns])


def read_stats(path: Path, columns: list) -> list:
    """Читает статистику запусков, возвращает список dict с ключами из columns."""
    if not path.exists():
        return []
    to_key = {label: key for key, label in columns}
    with path.open(encoding="utf-8-sig") as f:
        return [{to_key.get(k, k): v for k, v in row.items()}
                for row in csv.DictReader(f, delimiter=";")]


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
            w = csv.writer(f, delimiter=";")
            w.writerow([label for _, label in CSV_COLUMNS])
            for e in actual:
                row = {
                    **e,
                    "sources_count": len(e["sources"]),
                    "sources": " ".join(s["post_link"] for s in e["sources"]),
                }
                w.writerow([row.get(key, "") for key, _ in CSV_COLUMNS])
    except PermissionError:
        log.warning(f"Не удалось записать {OUTPUT_FILE.name}: закройте его в Excel и запустите снова. "
                    f"Данные сохранены в {EVENTS_FILE.name}.")
    return len(actual)
