"""
Парсер IT-соревнований (хакатоны, олимпиады, конкурсы, CTF, кейс-чемпионаты...)
из публичных Telegram-каналов.

Как работает:
  1. Telethon забирает новые посты из каналов channels.txt.
  2. Предфильтр без API отсекает мусор: короткие посты, посты без слов-маркеров
     соревнований, одинаковые тексты и посты, которые уже есть в базе (правило A).
  3. Оставшиеся посты разбирает LLM (DeepSeek через OpenAI-совместимый API)
     и возвращает JSON: соревнование ли это, название, даты, формат, ссылка...
  4. Прошедшие события отбрасываются. Дубли склеиваются с уже известными событиями:
     по ссылке на регистрацию (правило B) и по дате + похожему названию (правило C).
  5. Всё хранится в events.json, актуальные события выгружаются в events.csv,
     статистика запуска дописывается в logs/parser_runs.csv.

Запуск:  python parser.py
"""
import asyncio
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

from openai import APIError
from telethon import TelegramClient
from telethon.errors import FloodWaitError, UsernameInvalidError, UsernameNotOccupiedError

from common import (
    BASE, EVENTS_FILE, LLM_MODEL, LOGS_DIR, OUTPUT_FILE, LLMUsage, append_stats, clean_date,
    clean_str, export_csv, find_duplicate, is_actual, known_post_links, llm_json, load_events,
    make_llm, merge_into, next_id, pct, save_events, setup_logging,
)

API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
DAYS_BACK = int(os.getenv("DAYS_BACK", "30"))       # сколько дней назад смотреть при первом запуске
LIMIT_PER_CHANNEL = int(os.getenv("LIMIT_PER_CHANNEL", "300"))
ONLY_ONLINE = os.getenv("ONLY_ONLINE", "0") == "1"  # 1 = только онлайн/гибрид, 0 = все
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "150"))  # более короткие посты не отправляем в LLM
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "5"))  # сколько запросов к LLM одновременно
MAX_POST_CHARS = 4000  # длинные посты обрезаем, чтобы не платить за лишние токены

CHANNELS_FILE = BASE / "channels.txt"
STATE_FILE = BASE / "state.json"   # последний прочитанный id по каждому каналу
STATS_FILE = LOGS_DIR / "parser_runs.csv"

# Предфильтр: в посте должно быть хотя бы одно слово, связанное с соревнованиями.
# Список намеренно широкий, точную классификацию делает LLM.
COMPETITION_RE = re.compile(
    r"хакатон|hackathon|олимпиад|olymp|конкурс|contest|чемпионат|championship|"
    r"соревнован|competition|\bctf\b|турнир|tournament|\bjam\b|джем|кейс|датафон|"
    r"datathon|challenge|челлендж|kaggle|кубок|\bcup\b|баттл|battle",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """\
Ты разбираешь посты из Telegram-каналов и ищешь анонсы IT-соревнований.

Соревнование — это мероприятие, где человек или команда решает задачи и результаты \
оцениваются: хакатон, олимпиада, конкурс, CTF, кейс-чемпионат, game jam, датафон, \
соревнование по программированию/ML/анализу данных/кибербезопасности/робототехнике, \
отбор в акселератор с решением задач и т.п.

НЕ соревнование: вебинар, митап, конференция, лекция, курс, интенсив без конкурсной \
части, вакансия, стажировка, реклама, новость. Также НЕ считай соревнованием пост, \
который только подводит итоги, объявляет победителей или разбирает задачи уже \
прошедшего соревнования, — нам нужны только анонсы, где ещё можно участвовать \
или следить за событием.

Даты пиши в формате YYYY-MM-DD. Если год не указан, определи его по дате публикации \
поста (анонс в декабре про январь — следующий год). Если дату не знаешь — null. \
Ничего не выдумывай: указывай только то, что есть в тексте.

Ответь строго одним JSON-объектом с ключами:
{
  "is_competition": true/false,
  "kind": "hackathon" | "olympiad" | "contest" | "ctf" | "case" | "gamejam" | "other",
  "title": "короткое название соревнования",
  "summary": "одно предложение: что нужно делать участникам",
  "start_date": "YYYY-MM-DD" | null,
  "end_date": "YYYY-MM-DD" | null,
  "registration_deadline": "YYYY-MM-DD" | null,
  "format": "online" | "offline" | "hybrid" | "unknown",
  "city": "город проведения" | null,
  "participation": "solo" | "team" | "any" | "unknown",
  "audience": "для кого: школьники, студенты, все и т.п." | null,
  "prize": "призовой фонд или призы" | null,
  "registration_url": "ссылка на регистрацию или страницу соревнования" | null
}
Если is_competition = false, остальные поля можно заполнить null.
"""

STATS_FIELDS = [
    "started_at", "status", "error", "duration_s", "model", "channels", "channels_failed",
    "posts_read", "dropped_short", "dropped_no_keywords", "dropped_same_text", "dup_post_link",
    "sent_to_llm", "llm_failed", "llm_error_pct", "llm_requests", "llm_bad_json",
    "prompt_tokens", "completion_tokens",
    "not_competition", "past", "dropped_not_online", "dup_url", "dup_title",
    "new_events", "events_total", "events_actual",
]


def text_key(text: str) -> str:
    """Ключ для поиска одинаковых постов (репосты одного анонса в разных каналах)."""
    return re.sub(r"\W+", "", text.lower())[:300]


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_channels():
    channels = []
    for line in CHANNELS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        line = line.replace("https://t.me/", "").replace("t.me/", "").lstrip("@").strip("/")
        channels.append(line)
    return channels


async def fetch_candidates(client, channel, last_id, since, seen, known_links, stats, log):
    """Читает новые посты канала, применяет предфильтр. Возвращает (кандидаты для LLM, макс. id)."""
    posts = []
    max_id_seen = last_id
    total = 0
    async for msg in client.iter_messages(channel, limit=LIMIT_PER_CHANNEL, min_id=last_id):
        max_id_seen = max(max_id_seen, msg.id)
        if not last_id and msg.date < since:
            break
        total += 1
        link = f"https://t.me/{channel}/{msg.id}"
        text = (msg.text or "").strip()
        if len(text) < MIN_TEXT_LEN:
            stats["dropped_short"] += 1
            continue
        if not COMPETITION_RE.search(text):
            stats["dropped_no_keywords"] += 1
            continue
        if link in known_links:  # правило A
            stats["dup_post_link"] += 1
            log.debug(f"уже в базе: {link}")
            continue
        key = text_key(text)
        if key in seen:
            stats["dropped_same_text"] += 1
            log.debug(f"одинаковый текст: {link}")
            continue
        seen.add(key)
        posts.append({
            "channel": channel,
            "post_date": msg.date.strftime("%Y-%m-%d %H:%M"),
            "post_link": link,
            "text": text,
        })
    stats["posts_read"] += total
    log.info(f"@{channel}: постов {total}, в LLM пойдёт {len(posts)}")
    return posts, max_id_seen


async def classify(llm, sem, post, today, usage):
    user_msg = (
        f"Сегодня: {today}\n"
        f"Дата публикации поста: {post['post_date']}\n"
        f"Канал: @{post['channel']}\n\n"
        f"Текст поста:\n{post['text'][:MAX_POST_CHARS]}"
    )
    async with sem:
        return await llm_json(llm, SYSTEM_PROMPT, user_msg, usage)


def to_event(data: dict, post: dict, event_id: int) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": event_id,
        "start_date": clean_date(data.get("start_date")),
        "end_date": clean_date(data.get("end_date")),
        "registration_deadline": clean_date(data.get("registration_deadline")),
        "kind": clean_str(data.get("kind")),
        "title": clean_str(data.get("title"))[:200],
        "summary": clean_str(data.get("summary"))[:500],
        "format": clean_str(data.get("format")),
        "city": clean_str(data.get("city")),
        "participation": clean_str(data.get("participation")),
        "audience": clean_str(data.get("audience")),
        "prize": clean_str(data.get("prize")),
        "registration_url": clean_str(data.get("registration_url")),
        "sources": [{k: post[k] for k in ("post_link", "channel", "post_date")}],
        "created_at": now,
        "updated_at": now,
    }


async def run(stats, log):
    if not API_ID or not API_HASH:
        raise SystemExit("Заполните API_ID и API_HASH в файле .env (пример в .env.example)")
    channels = load_channels()
    if not channels:
        raise SystemExit("Добавьте каналы в channels.txt")
    llm = make_llm()

    state = load_state()
    new_state = dict(state)
    events = load_events()
    since = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    today = date.today().isoformat()
    stats["channels"] = len(channels)

    # 1. Telegram + предфильтр
    candidates = []
    seen = set()
    known_links = known_post_links(events)
    async with TelegramClient(str(BASE / "session"), API_ID, API_HASH) as client:
        for ch in channels:
            try:
                posts, max_id = await fetch_candidates(
                    client, ch, state.get(ch, 0), since, seen, known_links, stats, log)
                new_state[ch] = max_id
                candidates.extend(posts)
            except FloodWaitError as e:
                stats["channels_failed"] += 1
                log.warning(f"@{ch}: FloodWait, ждём {e.seconds} сек, канал пропущен в этом запуске")
                await asyncio.sleep(e.seconds + 5)
            except (UsernameInvalidError, UsernameNotOccupiedError, ValueError):
                stats["channels_failed"] += 1
                log.warning(f"@{ch}: канал не найден, пропускаю")
            except Exception as e:  # noqa
                stats["channels_failed"] += 1
                log.warning(f"@{ch}: ошибка {type(e).__name__}: {e}")
            await asyncio.sleep(2)  # пауза между каналами, чтобы не словить FloodWait

    # 2. LLM. Старые посты первыми, чтобы основным источником события был самый ранний анонс
    candidates.sort(key=lambda p: p["post_date"])
    stats["sent_to_llm"] = len(candidates)
    log.info(f"\nОтправляю в {LLM_MODEL}: {len(candidates)} постов...")
    usage = LLMUsage()
    sem = asyncio.Semaphore(LLM_CONCURRENCY)
    try:
        results = await asyncio.gather(*(classify(llm, sem, p, today, usage) for p in candidates))
    finally:
        stats.update(llm_requests=usage.requests, llm_bad_json=usage.bad_json,
                     prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)

    # 3. Отбор и склейка дублей
    for post, data in zip(candidates, results):
        link = post["post_link"]
        if data is None:
            stats["llm_failed"] += 1
            log.warning(f"  не удалось разобрать ответ модели: {link}")
            continue
        event = to_event(data, post, next_id(events))
        if not data.get("is_competition"):
            stats["not_competition"] += 1
            log.debug(f"не соревнование: {link} | {event['title']}")
            continue
        if not is_actual(event, today):
            stats["past"] += 1
            log.debug(f"уже прошло: {link} | {event['title']}")
            continue
        if ONLY_ONLINE and event["format"] not in ("online", "hybrid"):
            stats["dropped_not_online"] += 1
            log.debug(f"не онлайн: {link} | {event['title']}")
            continue
        dup, rule = find_duplicate(events, event)
        if dup:
            stats[f"dup_{rule}"] += 1
            merge_into(dup, event)
            log.info(f"  = дубль ({rule}) #{dup['id']} {dup['title']}  <- {link}")
            continue
        events.append(event)
        stats["new_events"] += 1
        log.info(f"  + #{event['id']} {event['start_date'] or '????-??-??'}  {event['title']}  ({link})")

    save_events(events)
    stats["events_total"] = len(events)
    stats["events_actual"] = export_csv(events, log)
    # state сохраняем последним: если что-то упало раньше, посты обработаются при следующем запуске
    save_state(new_state)


def main():
    log = setup_logging("parser")
    started = time.monotonic()
    stats = {f: 0 for f in STATS_FIELDS}
    stats.update(started_at=datetime.now().isoformat(timespec="seconds"),
                 status="ok", error="", model=LLM_MODEL)
    log.debug("=== запуск parser.py ===")
    try:
        asyncio.run(run(stats, log))
    except APIError as e:
        # state.json не сохраняем: при следующем запуске эти посты обработаются заново
        stats.update(status="aborted", error=f"LLM API: {e}"[:300])
        log.error(f"Ошибка LLM API, запуск прерван, ничего не записано: {e}")
    except (Exception, KeyboardInterrupt) as e:
        stats.update(status="aborted", error=f"{type(e).__name__}: {e}"[:300])
        raise
    finally:
        stats["duration_s"] = round(time.monotonic() - started)
        stats["llm_error_pct"] = pct(stats["llm_failed"], stats["sent_to_llm"])
        append_stats(STATS_FILE, STATS_FIELDS, stats)

    log.info(
        f"\nГотово за {stats['duration_s']} с. Прочитано постов: {stats['posts_read']}, "
        f"в LLM: {stats['sent_to_llm']}, ошибок модели: {stats['llm_failed']} ({stats['llm_error_pct']}%).\n"
        f"Не соревнование: {stats['not_competition']}, прошли: {stats['past']}, "
        f"дубли: ссылка на пост {stats['dup_post_link']}, одинаковый текст {stats['dropped_same_text']}, "
        f"ссылка на регистрацию {stats['dup_url']}, название+дата {stats['dup_title']}.\n"
        f"Новых событий: {stats['new_events']}, всего в базе: {stats['events_total']}, "
        f"актуальных в {OUTPUT_FILE.name}: {stats['events_actual']}.\n"
        f"Статистика: {STATS_FILE.relative_to(BASE)}, подробный лог: logs/parser.log"
    )


if __name__ == "__main__":
    main()
