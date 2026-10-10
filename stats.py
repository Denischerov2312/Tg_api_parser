"""
Сводка по запускам parser.py и dedup_llm.py в виде таблиц в терминале.

Запуск:
  python stats.py          последние 10 запусков + сводка по моделям
  python stats.py -n 30    последние 30 запусков
"""
import argparse
from collections import defaultdict

from common import (
    DEDUP_STATS_COLUMNS, DEDUP_STATS_FILE, PARSER_STATS_COLUMNS, PARSER_STATS_FILE, pct,
    read_stats,
)


def num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


def fmt(value) -> str:
    """Целые без .0, дробные с двумя знаками."""
    v = num(value)
    return str(int(v)) if v == int(v) else str(round(v, 2))


def short_model(model: str) -> str:
    return model.split("/")[-1]


def print_table(headers: list, rows: list):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(f"{str(x):<{w}}" for x, w in zip(r, widths)))


def show_parser(runs: list, n: int):
    print(f"\n=== Последние запуски parser.py (всего запусков: {len(runs)}) ===\n")
    if not runs:
        print("Запусков ещё не было.")
        return
    headers = ["Дата", "Статус", "Модель", "Прочитано", "В модель", "Ошибок модели",
               "Не соревн.", "Прошли", "Дубли", "Новых", "Токенов", "Стоимость", "Время, с"]
    rows = []
    for r in runs[-n:]:
        rows.append([
            r["started_at"][:16],
            "OK" if r["status"] == "ok" else "ПРЕРВАН",
            short_model(r["model"]),
            fmt(r["posts_read"]),
            fmt(r["sent_to_llm"]),
            f"{fmt(r['llm_failed'])} ({fmt(r['llm_error_pct'])}%)",
            fmt(r["not_competition"]),
            fmt(r["past"]),
            fmt(num(r["dup_post_link"]) + num(r["dropped_same_text"]) + num(r["dup_url"]) + num(r["dup_title"])),
            fmt(r["new_events"]),
            fmt(num(r["prompt_tokens"]) + num(r["completion_tokens"])),
            fmt(r["cost"]) if r["cost"] else "—",
            fmt(r["duration_s"]),
        ])
    print_table(headers, rows)
    print("\nДубли = уже в базе + повтор текста + ссылка на регистрацию + название и дата.")

    aborted = [r for r in runs[-n:] if r["status"] != "ok"]
    for r in aborted:
        print(f"ПРЕРВАН {r['started_at'][:16]}: {r['error']}")

    # Сводка по моделям: только успешные запуски, где что-то отправлялось в модель
    by_model = defaultdict(lambda: defaultdict(float))
    for r in runs:
        if r["status"] != "ok" or not num(r["sent_to_llm"]):
            continue
        m = by_model[r["model"]]
        m["runs"] += 1
        for key in ("sent_to_llm", "llm_failed", "not_competition", "prompt_tokens",
                    "completion_tokens", "cost", "new_events"):
            m[key] += num(r[key])
    if not by_model:
        return
    print("\n=== Сводка по моделям (parser.py, все успешные запуски) ===\n")
    headers = ["Модель", "Запусков", "Постов в модель", "Ошибок модели, %",
               "Доля соревнований, %", "Токенов на пост", "Новых событий", "Стоимость всего"]
    rows = []
    for model, m in by_model.items():
        answered = m["sent_to_llm"] - m["llm_failed"]
        rows.append([
            model,
            fmt(m["runs"]),
            fmt(m["sent_to_llm"]),
            fmt(pct(m["llm_failed"], m["sent_to_llm"])),
            fmt(pct(answered - m["not_competition"], answered)),
            fmt(round((m["prompt_tokens"] + m["completion_tokens"]) / m["sent_to_llm"])),
            fmt(m["new_events"]),
            fmt(m["cost"]) if m["cost"] else "—",
        ])
    print_table(headers, rows)


def show_dedup(runs: list, n: int):
    print(f"\n=== Последние запуски dedup_llm.py (всего запусков: {len(runs)}) ===\n")
    if not runs:
        print("Запусков ещё не было.")
        return
    headers = ["Дата", "Статус", "Модель", "Режим", "Проверено", "Групп", "Некорректных",
               "Склеено", "Токенов", "Стоимость"]
    rows = []
    for r in runs[-n:]:
        rows.append([
            r["started_at"][:16],
            "OK" if r["status"] == "ok" else "ПРЕРВАН",
            short_model(r["model"]),
            "пробный" if r["dry_run"] == "1" else "склейка",
            fmt(r["events_checked"]),
            fmt(r["groups_returned"]),
            f"{fmt(r['groups_invalid'])} ({fmt(r['llm_error_pct'])}%)",
            fmt(r["events_merged"]),
            fmt(num(r["prompt_tokens"]) + num(r["completion_tokens"])),
            fmt(r["cost"]) if r["cost"] else "—",
        ])
    print_table(headers, rows)


def main():
    parser = argparse.ArgumentParser(description="Сводка по запускам парсера и поиска дублей")
    parser.add_argument("-n", type=int, default=10, help="сколько последних запусков показать (по умолчанию 10)")
    args = parser.parse_args()

    parser_runs = read_stats(PARSER_STATS_FILE, PARSER_STATS_COLUMNS)
    show_parser(parser_runs, args.n)
    show_dedup(read_stats(DEDUP_STATS_FILE, DEDUP_STATS_COLUMNS), args.n)
    if parser_runs:
        print(f"\nПодробный лог последнего запуска парсера: {parser_runs[-1].get('log_file', '—')}")
    print(f"Полная статистика для Excel: {PARSER_STATS_FILE.name}, {DEDUP_STATS_FILE.name} в папке logs")


if __name__ == "__main__":
    main()
