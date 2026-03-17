from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_path(path_parts: list[Any]) -> str:
    if not path_parts:
        return "<root>"

    result = ""
    for part in path_parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            if result and not result.endswith("]"):
                result += "."
            result += str(part)
    return result


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, list):
        return sorted((normalize(v) for v in value), key=repr)
    return value


def compare(
        a: Any,
        b: Any,
        path: list[Any] | None = None,
        only_in_a: list[str] | None = None,
        only_in_b: list[str] | None = None,
        changed_values: list[str] | None = None,
        changed_lists: list[str] | None = None,
        ignore_order: bool = False,
) -> dict[str, list[str]]:
    if path is None:
        path = []
    if only_in_a is None:
        only_in_a = []
    if only_in_b is None:
        only_in_b = []
    if changed_values is None:
        changed_values = []
    if changed_lists is None:
        changed_lists = []

    if isinstance(a, dict) and isinstance(b, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())

        for key in sorted(a_keys - b_keys, key=str):
            only_in_a.append(f"{format_path(path + [key])}: {a[key]!r}")

        for key in sorted(b_keys - a_keys, key=str):
            only_in_b.append(f"{format_path(path + [key])}: {b[key]!r}")

        for key in sorted(a_keys & b_keys, key=str):
            compare(
                a[key],
                b[key],
                path + [key],
                only_in_a,
                only_in_b,
                changed_values,
                changed_lists,
                ignore_order,
            )

    elif isinstance(a, list) and isinstance(b, list):
        path_str = format_path(path)

        if ignore_order:
            norm_a = normalize(a)
            norm_b = normalize(b)
            if norm_a != norm_b:
                changed_lists.append(
                    f"{path_str}:\n"
                    f"  A = {a!r}\n"
                    f"  B = {b!r}"
                )
        else:
            if a != b:
                changed_lists.append(
                    f"{path_str}:\n"
                    f"  A = {a!r}\n"
                    f"  B = {b!r}"
                )

    else:
        if a != b:
            changed_values.append(
                f"{format_path(path)}:\n"
                f"  A = {a!r}\n"
                f"  B = {b!r}"
            )

    return {
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "changed_values": changed_values,
        "changed_lists": changed_lists,
    }


def print_section(title: str, items: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not items:
        print("None")
    else:
        for item in items:
            print(item)


def print_diffs(file_a, file_b, ignore_order):
    if not Path(file_a).exists():
        print(f"File not found: {file_a}")
        raise SystemExit(1)

    if not Path(file_b).exists():
        print(f"File not found: {file_b}")
        raise SystemExit(1)

    try:
        data_a = load_yaml(file_a)
        data_b = load_yaml(file_b)
    except yaml.YAMLError as e:
        print(f"YAML parse error: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"Failed to read file: {e}")
        raise SystemExit(1)

    results = compare(data_a, data_b, ignore_order=ignore_order)

    total_diffs = sum(len(v) for v in results.values())
    print(f"Total differences found: {total_diffs}")

    print_section("ONLY IN A", results["only_in_a"])
    print_section("ONLY IN B", results["only_in_b"])
    print_section("CHANGED VALUES", results["changed_values"])
    print_section("CHANGED LISTS", results["changed_lists"])


if __name__ == "__main__":
    # file_a = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\steps_40k\2026-03-06\2026-03-06_T_12-14-07\2026-03-06_T_12-14-07_config.yaml"
    file_a = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\steps_100k\2026-03-06\2026-03-06_T_16-48-27\2026-03-06_T_16-48-27_config.yaml"
    file_b = r"C:\Users\saar.nehemia\PycharmProjects\Bereshit\outputs\steps_155k\2026-03-08\2026-03-08_T_06-41-04\2026-03-08_T_06-41-04_config.yaml"
    ignore_order = False

    print_diffs(file_a, file_b, ignore_order)
