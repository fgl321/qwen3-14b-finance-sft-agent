import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.routing import APIRoute
from starlette.routing import Mount, Route, WebSocketRoute

from app.main import app


def main() -> None:
    print("=" * 100)
    print("FastAPI 已注册路由详细列表")
    print("=" * 100)

    print(f"route_count = {len(app.routes)}")
    print("=" * 100)

    for index, route in enumerate(app.routes, start=1):
        route_type = type(route).__name__

        path = getattr(route, "path", None)
        name = getattr(route, "name", None)
        methods = getattr(route, "methods", None)
        endpoint = getattr(route, "endpoint", None)

        endpoint_name = getattr(endpoint, "__name__", None)

        print(f"[{index}] type={route_type}")
        print(f"    path={path!r}")
        print(f"    name={name!r}")
        print(f"    methods={methods!r}")
        print(f"    endpoint={endpoint_name!r}")

        if isinstance(route, APIRoute):
            print(f"    tags={route.tags!r}")
            print(f"    response_model={route.response_model!r}")

        if isinstance(route, WebSocketRoute):
            print("    websocket=True")

        if isinstance(route, Mount):
            print(f"    mount_routes_count={len(route.routes)}")

        print("-" * 100)


if __name__ == "__main__":
    main()
