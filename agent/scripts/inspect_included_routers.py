import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


def main() -> None:
    print("=" * 100)
    print("检查 _IncludedRouter 对象")
    print("=" * 100)

    for index, route in enumerate(app.routes, start=1):
        route_type = type(route).__name__

        if route_type != "_IncludedRouter":
            continue

        print(f"\n[{index}] type={route_type}")
        print("-" * 100)

        print("repr:")
        print(repr(route))

        print("\n__dict__:")
        pprint(getattr(route, "__dict__", {}))

        print("\ndir 中疑似关键字段:")
        names = [
            name
            for name in dir(route)
            if not name.startswith("__")
        ]

        interesting_names = [
            name
            for name in names
            if any(
                keyword in name.lower()
                for keyword in [
                    "router",
                    "route",
                    "path",
                    "prefix",
                    "method",
                    "endpoint",
                    "target",
                    "app",
                ]
            )
        ]

        pprint(interesting_names)

        print("-" * 100)


if __name__ == "__main__":
    main()
