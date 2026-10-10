"""
Поиск дублей событий с помощью LLM (запускается вручную, когда нужно).

parser.py склеивает только очевидные дубли (одинаковая ссылка на регистрацию или
та же дата + похожее название). Этот скрипт отправляет модели список актуальных
событий из events.json и просит найти группы записей об одном и том же событии,
даже если у них разные названия или даты (например, анонс и напоминание).

Запуск:
  python dedup_llm.py            найти и склеить дубли
  python dedup_llm.py --dry-run  только показать, что модель предлагает склеить

Перед изменением events.json делается копия events.json.bak.
Статистика запуска дописывается в logs/dedup_runs.csv.
"""
import argparse
import asyncio
import json
import shutil
import time
from datetime import date, datetime

from openai import APIError

from common import (
    BASE, EVENTS_FILE, LLM_MODEL, LOGS_DIR, LLMUsage, append_stats, export_csv, is_actual,
    llm_json, load_events, make_llm, merge_into, pct, save_events, setup_logging,
)

MAX_EVENTS = 150  # больше событий в одном запросе не отправляем, чтобы модель не теряла внимание
STATS_FILE = LOGS_DIR / "dedup_runs.csv"

SYSTEM_PROMPT = """\
Тебе дан список IT-соревнований (хакатоны, олимпиады, конкурсы, CTF и т.п.), \
собранных из разных Telegram-постов. Некоторые записи описывают одно и то же \
соревнование: один анонс в разных каналах, анонс и напоминание, пост про \
регистрацию и пост про финал.

Найди группы записей об ОДНОМ И ТОМ ЖЕ соревновании.
- Разные этапы одного соревнования (отборочный этап и финал, онлайн-отбор и очный \
финал) считай одним соревнованием.
- Разные соревнования одного организатора или одной серии (например, два разных \
CTF в один день, разные сезоны или треки с отдельной регистрацией) — разные записи.
- Объединяй только если уверен. Если сомневаешься — не объединяй.

Ответь строго одним JSON-объектом:
{"groups": [{"ids": [<id>, <id>, ...], "reason": "почему это одно соревнование"}]}
Каждая группа — минимум 2 id из списка, один id может входить только в одну группу. \
Если дублей нет, верни {"groups": []}.
"""

STATS_FIELDS = [
    "started_at", "status", "error", "duration_s", "model", "dry_run", "events_checked",
    "groups_returned", "groups_invalid", "llm_error_pct", "llm_requests", "llm_bad_json",
    "prompt_tokens", "completion_tokens", "events_merged", "events_total_after",
]


def event_line(e: dict) -> str:
    short = {
        "id": e["id"], "kind": e["kind"], "title": e["title"],
        "start": e["start_date"] or None, "end": e["end_date"] or None,
        "deadline": e["registration_deadline"] or None, "city": e["city"] or None,
        "url": e["registration_url"] or None, "summary": e["summary"][:200],
    }
    return json.dumps(short, ensure_ascii=False)


def validate_groups(data, valid_ids: set, log) -> tuple:
    """Оставляет только корректные группы. Возвращает (группы, сколько всего вернула модель)."""
    raw = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return [], 0
    groups, used = [], set()
    for g in raw:
        ids = g.get("ids") if isinstance(g, dict) else None
        ok = (isinstance(ids, list) and len(set(ids)) >= 2
              and all(isinstance(i, int) and i in valid_ids for i in ids)
              and not used & set(ids))
        if not ok:
            log.warning(f"  некорректная группа от модели, пропускаю: {g}")
            continue
        used |= set(ids)
        groups.append({"ids": sorted(set(ids)), "reason": str(g.get("reason", ""))})
    return groups, len(raw)


async def run(stats, dry_run, log):
    events = load_events()
    today = date.today().isoformat()
    actual = [e for e in events if is_actual(e, today)]
    actual.sort(key=lambda e: e["start_date"] or e["registration_deadline"] or "9999")
    if len(actual) > MAX_EVENTS:
        log.warning(f"Актуальных событий {len(actual)}, проверяю ближайшие {MAX_EVENTS}")
        actual = actual[:MAX_EVENTS]
    stats["events_checked"] = len(actual)
    if len(actual) < 2:
        log.info("Проверять нечего: актуальных событий меньше двух.")
        stats["events_total_after"] = len(events)
        return

    log.info(f"Отправляю в {LLM_MODEL} список из {len(actual)} событий...")
    usage = LLMUsage()
    user_msg = f"Сегодня: {today}\n\nСобытия (по одному JSON на строку):\n" + \
        "\n".join(event_line(e) for e in actual)
    try:
        data = await llm_json(make_llm(), SYSTEM_PROMPT, user_msg, usage)
    finally:
        stats.update(llm_requests=usage.requests, llm_bad_json=usage.bad_json,
                     prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)

    if data is None:
        stats["llm_error_pct"] = 100.0
        stats["events_total_after"] = len(events)
        log.error("Модель не вернула корректный JSON, ничего не изменено.")
        return

    by_id = {e["id"]: e for e in events}
    groups, returned = validate_groups(data, {e["id"] for e in actual}, log)
    stats["groups_returned"] = returned
    stats["groups_invalid"] = returned - len(groups)
    stats["llm_error_pct"] = pct(stats["groups_invalid"], returned)

    if not groups:
        log.info("Модель не нашла дублей.")
    for g in groups:
        main_id, *others = g["ids"]  # основным остаётся событие, найденное раньше всех
        log.info(f"\n#{main_id} {by_id[main_id]['title']}")
        for i in others:
            log.info(f"  <- #{i} {by_id[i]['title']}")
        log.info(f"  причина: {g['reason']}")
        if not dry_run:
            for i in others:
                merge_into(by_id[main_id], by_id.pop(i))
                stats["events_merged"] += 1

    events = list(by_id.values())
    stats["events_total_after"] = len(events)
    if dry_run:
        log.info("\n--dry-run: ничего не изменено.")
        return
    if stats["events_merged"]:
        shutil.copy(EVENTS_FILE, EVENTS_FILE.with_name(EVENTS_FILE.name + ".bak"))
        save_events(events)
        export_csv(events, log)


def main():
    parser = argparse.ArgumentParser(description="Поиск дублей событий с помощью LLM")
    parser.add_argument("--dry-run", action="store_true", help="только показать, ничего не менять")
    args = parser.parse_args()

    log = setup_logging("dedup")
    started = time.monotonic()
    stats = {f: 0 for f in STATS_FIELDS}
    stats.update(started_at=datetime.now().isoformat(timespec="seconds"), status="ok",
                 error="", model=LLM_MODEL, dry_run=int(args.dry_run))
    log.debug(f"=== запуск dedup_llm.py{' --dry-run' if args.dry_run else ''} ===")
    try:
        asyncio.run(run(stats, args.dry_run, log))
    except APIError as e:
        stats.update(status="aborted", error=f"LLM API: {e}"[:300])
        log.error(f"Ошибка LLM API, ничего не изменено: {e}")
    except (Exception, KeyboardInterrupt) as e:
        stats.update(status="aborted", error=f"{type(e).__name__}: {e}"[:300])
        raise
    finally:
        stats["duration_s"] = round(time.monotonic() - started)
        append_stats(STATS_FILE, STATS_FIELDS, stats)

    log.info(
        f"\nГотово за {stats['duration_s']} с. Проверено событий: {stats['events_checked']}, "
        f"групп дублей: {stats['groups_returned']} (некорректных {stats['groups_invalid']}), "
        f"склеено записей: {stats['events_merged']}.\n"
        f"Статистика: {STATS_FILE.relative_to(BASE)}, лог: logs/dedup.log"
    )


if __name__ == "__main__":
    main()
