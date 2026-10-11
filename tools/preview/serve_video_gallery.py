"""Serve a chosen local gallery directory with byte ranges for video seeking."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

import range_http_server_8766 as range_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8778)
    args = parser.parse_args()
    range_server.ROOT = str(args.root.resolve(strict=True))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), range_server.RangeHandler)
    print(f'Video gallery: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
