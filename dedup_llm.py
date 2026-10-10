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
Статистика запуска дописывается в logs/dedup_runs.csv,
подробный лог — в logs/runs/<дата_время>_dedup.log.
"""
import argparse
import asyncio
import json
import shutil
import time
from datetime import date, datetime

from openai import APIError

from common import (
    BASE, DEDUP_STATS_COLUMNS, DEDUP_STATS_FILE, EVENTS_FILE, LLM_MODEL, LLMUsage, append_stats,
    export_csv, is_actual, llm_cost, llm_json, load_events, make_llm, merge_into, pct,
    save_events, setup_logging, tag,
)

MAX_EVENTS = 150  # больше событий в одном запросе не отправляем, чтобы модель не теряла внимание

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
            log.warning(tag("ошибка модели", f"некорректная группа, пропускаю: {g}"))
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
    stats["events_total_after"] = len(events)
    if len(actual) < 2:
        log.info("Проверять нечего: актуальных событий меньше двух.")
        return

    log.debug("Отправленный список событий:\n" + "\n".join(
        f"  #{e['id']} {e['start_date'] or '?'}  {e['title']}" for e in actual))
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
        log.error(tag("ошибка модели", "модель не вернула корректный JSON, ничего не изменено"))
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
        log.info("")
        log.info(tag("ДУБЛИ" if not dry_run else "ДУБЛИ (пробно)", f"#{main_id} {by_id[main_id]['title']}"))
        for i in others:
            log.info(f"{'':<24}   <- #{i} {by_id[i]['title']}")
        log.info(f"{'':<24} причина: {g['reason']}")
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


def summary(s: dict) -> str:
    cost = f", стоимость {s['cost']}" if s["cost"] != "" else ""
    status = "OK" if s["status"] == "ok" else f"ПРЕРВАН — {s['error']}"
    mode = "пробный (--dry-run), ничего не изменено" if s["dry_run"] else "склейка"
    return f"""
==================== ИТОГ ПОИСКА ДУБЛЕЙ ====================
Статус:               {status}
Режим:                {mode}
Модель:               {s['model']}
Проверено событий:    {s['events_checked']}
Групп дублей:         {s['groups_returned']} (некорректных: {s['groups_invalid']}, {s['llm_error_pct']}%)
Токенов:              {s['prompt_tokens']} на вход, {s['completion_tokens']} на выход{cost}
Склеено записей:      {s['events_merged']}
В базе после:         {s['events_total_after']}
Время:                {s['duration_s']} с
Подробный лог:        {s['log_file']}
Статистика запусков:  {DEDUP_STATS_FILE.relative_to(BASE)}  (или python stats.py)
============================================================"""


def main():
    parser = argparse.ArgumentParser(description="Поиск дублей событий с помощью LLM")
    parser.add_argument("--dry-run", action="store_true", help="только показать, ничего не менять")
    args = parser.parse_args()

    log, log_path = setup_logging("dedup")
    started = time.monotonic()
    stats = {key: 0 for key, _ in DEDUP_STATS_COLUMNS}
    stats.update(started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status="ok", error="",
                 model=LLM_MODEL, dry_run=int(args.dry_run), log_file=str(log_path.relative_to(BASE)))
    log.debug(f"Запуск dedup_llm.py{' --dry-run' if args.dry_run else ''} {stats['started_at']}")
    try:
        asyncio.run(run(stats, args.dry_run, log))
    except APIError as e:
        stats.update(status="aborted", error=f"LLM API: {e}"[:300])
        log.error(f"Ошибка LLM API, ничего не изменено: {e}")
    except BaseException as e:  # в т.ч. Ctrl+C и SystemExit — фиксируем в статистике и пробрасываем
        stats.update(status="aborted", error=f"{type(e).__name__}: {e}"[:300])
        raise
    finally:
        stats["duration_s"] = round(time.monotonic() - started)
        stats["cost"] = llm_cost(stats["prompt_tokens"], stats["completion_tokens"])
        append_stats(DEDUP_STATS_FILE, DEDUP_STATS_COLUMNS, stats)
        log.info(summary(stats))


if __name__ == "__main__":
    main()
