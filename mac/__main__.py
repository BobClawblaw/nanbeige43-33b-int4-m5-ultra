"""python3 -m mac build|up|down|status|logs|command|chat|validate|launchd"""

import argparse
import shlex
import sys

from . import config, engine, launchd, serve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m mac", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the pinned engine and libtfgrammar, under tf_cache")
    b.add_argument("--rebuild", action="store_true")
    up = sub.add_parser("up", help="start the server (in the background; --foreground for the launchd unit)")
    up.add_argument("--foreground", action="store_true")
    sub.add_parser("down", help="stop the server `up` started")
    sub.add_parser("status", help="pid, url and the server's health")
    logs = sub.add_parser("logs", help="the server log's last lines")
    logs.add_argument("-n", type=int, default=40)
    sub.add_parser("command", help="print the serve command")
    ch = sub.add_parser("chat", help="one prompt to the running server")
    ch.add_argument("prompt")
    ch.add_argument("--max-tokens", type=int, default=2048)
    sub.add_parser("validate", help="check the settings, the checkpoint and the build without starting anything")
    ld = sub.add_parser("launchd", help="write the launchd unit (not loaded) serving PORT to the LAN")
    ld.add_argument("--port", type=int, default=8888)
    a = p.parse_args(argv)
    s = config.load()
    if a.cmd == "build":
        engine.build(s, rebuild=a.rebuild)
        print(f"engine {s.run_binary}")
    elif a.cmd == "up":
        problems = config.validate(s)
        if problems:
            raise SystemExit("\n".join(problems))
        serve.up(s, foreground=a.foreground)
    elif a.cmd == "down":
        serve.down(s)
    elif a.cmd == "status":
        print(serve.status(s))
    elif a.cmd == "logs":
        print("\n".join(s.log_file.read_text(errors="replace").splitlines()[-a.n:]) if s.log_file.exists() else "no log")
    elif a.cmd == "command":
        print(shlex.join(s.serve_args()))
    elif a.cmd == "chat":
        print(serve.chat(s, a.prompt, a.max_tokens))
    elif a.cmd == "validate":
        problems = config.validate(s)
        print("\n".join(problems) if problems else f"ok: {s.served_name} from {s.model} with {s.run_binary} at {s.url}")
        return 1 if problems else 0
    elif a.cmd == "launchd":
        print(launchd.write(s, a.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
