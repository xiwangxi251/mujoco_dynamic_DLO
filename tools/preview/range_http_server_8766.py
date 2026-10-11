"""Small static server with HTTP Range support for browser video seeking."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from pathlib import Path
import os
import re

ROOT = r"C:\Users\27642\Desktop\mujoco\mujoco_dynamic_DLO\outputs\vertical_workspace\scripted_136_retest_20261004"

class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def send_head(self):
        path = self.translate_path(urlsplit(self.path).path)
        if not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        ctype = self.guess_type(path)
        self._range = None
        header = self.headers.get("Range")
        if header:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
            if not m:
                self.send_error(416, "Invalid Range")
                return None
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                length = int(m.group(2))
                start, end = max(0, size - length), size - 1
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            end = min(end, size - 1)
            self._range = (start, end)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(size))
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        return open(path, "rb")

    def copyfile(self, source, outputfile):
        if self._range is None:
            return super().copyfile(source, outputfile)
        start, end = self._range
        source.seek(start)
        remaining = end - start + 1
        while remaining:
            block = source.read(min(64 * 1024, remaining))
            if not block:
                break
            outputfile.write(block)
            remaining -= len(block)

if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8766), RangeHandler)
    print("Range HTTP server on http://127.0.0.1:8766", flush=True)
    server.serve_forever()
