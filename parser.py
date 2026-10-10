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
     статистика запуска дописывается в logs/parser_runs.csv,
     подробности по каждому посту — в logs/runs/<дата_время>_parser.log.

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
    BASE, LLM_MODEL, OUTPUT_FILE, PARSER_STATS_COLUMNS, PARSER_STATS_FILE, TITLE_SIMILARITY,
    LLMUsage, append_stats, clean_date, clean_str, export_csv, find_duplicate, is_actual,
    known_post_links, llm_cost, llm_json, load_events, make_llm, merge_into, next_id, pct,
    save_events, setup_logging, snippet, tag,
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
  "reason": "коротко, до 15 слов: почему это соревнование или нет",
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
Если is_competition = false, заполни is_competition и reason, остальные поля — null.
"""


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
    log.debug(f"\n--- @{channel} (последний прочитанный id: {last_id or 'нет, первый запуск'}) ---")
    async for msg in client.iter_messages(channel, limit=LIMIT_PER_CHANNEL, min_id=last_id):
        max_id_seen = max(max_id_seen, msg.id)
        if not last_id and msg.date < since:
            break
        total += 1
        link = f"https://t.me/{channel}/{msg.id}"
        text = (msg.text or "").strip()
        if len(text) < MIN_TEXT_LEN:
            stats["dropped_short"] += 1
            log.debug(tag("короткий", f"{link} | {snippet(text)}"))
            continue
        if not COMPETITION_RE.search(text):
            stats["dropped_no_keywords"] += 1
            log.debug(tag("нет ключевых слов", f"{link} | {snippet(text)}"))
            continue
        if link in known_links:  # правило A
            stats["dup_post_link"] += 1
            log.debug(tag("уже в базе", link))
            continue
        key = text_key(text)
        if key in seen:
            stats["dropped_same_text"] += 1
            log.debug(tag("повтор текста", f"{link} | {snippet(text)}"))
            continue
        seen.add(key)
        log.debug(tag("в модель", f"{link} | {snippet(text)}"))
        posts.append({
            "channel": channel,
            "post_date": msg.date.strftime("%Y-%m-%d %H:%M"),
            "post_link": link,
            "text": text,
        })
    stats["posts_read"] += total
    log.info(f"@{channel}: постов {total}, в модель пойдёт {len(posts)}")
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


def dates_str(e: dict) -> str:
    parts = [e["start_date"] or "?"]
    if e["end_date"] and e["end_date"] != e["start_date"]:
        parts.append(f"– {e['end_date']}")
    if e["registration_deadline"]:
        parts.append(f"(рег. до {e['registration_deadline']})")
    return " ".join(parts)


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
    log.debug(
        f"Настройки: модель {LLM_MODEL}, DAYS_BACK={DAYS_BACK}, LIMIT_PER_CHANNEL={LIMIT_PER_CHANNEL}, "
        f"MIN_TEXT_LEN={MIN_TEXT_LEN}, ONLY_ONLINE={int(ONLY_ONLINE)}, TITLE_SIMILARITY={TITLE_SIMILARITY}\n"
        f"Каналы: {', '.join('@' + c for c in channels)}\n"
        f"Событий в базе до запуска: {len(events)}"
    )

    # 1. Telegram + предфильтр
    log.debug("\n========== 1. ЧТЕНИЕ КАНАЛОВ И ПРЕДФИЛЬТР ==========")
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
                log.warning(tag("ошибка канала", f"@{ch}: FloodWait, ждём {e.seconds} сек, канал пропущен"))
                await asyncio.sleep(e.seconds + 5)
            except (UsernameInvalidError, UsernameNotOccupiedError, ValueError):
                stats["channels_failed"] += 1
                log.warning(tag("ошибка канала", f"@{ch}: канал не найден, пропускаю"))
            except Exception as e:  # noqa
                stats["channels_failed"] += 1
                log.warning(tag("ошибка канала", f"@{ch}: {type(e).__name__}: {e}"))
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
    log.debug("\n========== 2. РЕШЕНИЯ МОДЕЛИ И ДУБЛИ ==========")
    for post, data in zip(candidates, results):
        link = post["post_link"]
        if data is None:
            stats["llm_failed"] += 1
            log.warning(tag("ошибка модели", f"{link} — модель не вернула корректный JSON"))
            continue
        reason = clean_str(data.get("reason"))
        event = to_event(data, post, next_id(events))
        if not data.get("is_competition"):
            stats["not_competition"] += 1
            log.debug(tag("не соревнование", f"{link} | {reason} | {snippet(post['text'], 60)}"))
            continue
        if not is_actual(event, today):
            stats["past"] += 1
            log.debug(tag("уже прошло", f"{link} | {event['title']} | {dates_str(event)}"))
            continue
        if ONLY_ONLINE and event["format"] not in ("online", "hybrid"):
            stats["dropped_not_online"] += 1
            log.debug(tag("не онлайн", f"{link} | {event['title']} | {event['format']}"))
            continue
        dup, rule = find_duplicate(events, event)
        if dup:
            stats[f"dup_{rule}"] += 1
            merge_into(dup, event)
            label = "дубль: ссылка рег." if rule == "url" else "дубль: название+дата"
            log.info(tag(label, f"#{dup['id']} {dup['title']}  <-  {event['title']} | {link}"))
            continue
        events.append(event)
        stats["new_events"] += 1
        log.info(tag("НОВОЕ", f"#{event['id']} {event['title']} | {dates_str(event)} | {link}"))
        log.debug(f"{'':<24} {event['kind']}, {event['format']}, {event['city'] or 'город не указан'} | {reason}")

    save_events(events)
    stats["events_total"] = len(events)
    stats["events_actual"] = export_csv(events, log)
    # state сохраняем последним: если что-то упало раньше, посты обработаются при следующем запуске
    save_state(new_state)


def summary(s: dict) -> str:
    cost = f", стоимость {s['cost']}" if s["cost"] != "" else ""
    status = "OK" if s["status"] == "ok" else f"ПРЕРВАН — {s['error']}"
    return f"""
==================== ИТОГ ЗАПУСКА ====================
Статус:               {status}
Модель:               {s['model']}
Каналов:              {s['channels']} (с ошибкой: {s['channels_failed']})
Прочитано постов:     {s['posts_read']}
  отсеяно до модели:  короткие {s['dropped_short']}, без ключевых слов {s['dropped_no_keywords']}, \
повтор текста {s['dropped_same_text']}, уже в базе {s['dup_post_link']}
Отправлено в модель:  {s['sent_to_llm']}
  ошибок модели:      {s['llm_failed']} ({s['llm_error_pct']}%)
  токенов:            {s['prompt_tokens']} на вход, {s['completion_tokens']} на выход{cost}
  не соревнование:    {s['not_competition']}
  уже прошли:         {s['past']}
  не онлайн:          {s['dropped_not_online']}
Дубли склеены:        по ссылке на регистрацию {s['dup_url']}, по названию и дате {s['dup_title']}
Новых событий:        {s['new_events']}
В базе:               {s['events_total']}, актуальных в {OUTPUT_FILE.name}: {s['events_actual']}
Время:                {s['duration_s']} с
Подробный лог:        {s['log_file']}
Статистика запусков:  {PARSER_STATS_FILE.relative_to(BASE)}  (или python stats.py)
======================================================"""


def main():
    log, log_path = setup_logging("parser")
    started = time.monotonic()
    stats = {key: 0 for key, _ in PARSER_STATS_COLUMNS}
    stats.update(started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status="ok", error="",
                 model=LLM_MODEL, log_file=str(log_path.relative_to(BASE)))
    log.debug(f"Запуск parser.py {stats['started_at']}")
    try:
        asyncio.run(run(stats, log))
    except APIError as e:
        # state.json не сохраняем: при следующем запуске эти посты обработаются заново
        stats.update(status="aborted", error=f"LLM API: {e}"[:300])
        log.error(f"Ошибка LLM API, запуск прерван, ничего не записано: {e}")
    except BaseException as e:  # в т.ч. Ctrl+C и SystemExit — фиксируем в статистике и пробрасываем
        stats.update(status="aborted", error=f"{type(e).__name__}: {e}"[:300])
        raise
    finally:
        stats["duration_s"] = round(time.monotonic() - started)
        stats["llm_error_pct"] = pct(stats["llm_failed"], stats["sent_to_llm"])
        stats["cost"] = llm_cost(stats["prompt_tokens"], stats["completion_tokens"])
        append_stats(PARSER_STATS_FILE, PARSER_STATS_COLUMNS, stats)
        log.info(summary(stats))


if __name__ == "__main__":
    main()
