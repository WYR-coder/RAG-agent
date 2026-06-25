"""Reliable backend startup: kill anything holding port, retry, fallback."""
import os
import sys
import subprocess
import time
import socket

PREFERRED_PORT = 8080
MY_PID = str(os.getpid())

def free_port(port):
    """Kill process holding this specific port (not ourselves)."""
    print(f"[start] Checking port {port}...")
    try:
        r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=8)
        killed = False
        for line in r.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                if pid == MY_PID:
                    continue
                print(f"[start] Killing PID {pid} on port {port}")
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, timeout=8)
                killed = True
        if killed:
            time.sleep(2)
    except Exception as e:
        print(f"[start] Port check: {e}")

def port_is_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        return True
    except OSError:
        return False

# ── Free the preferred port ────────────────────────────────────────
free_port(PREFERRED_PORT)

# ── Find usable port ───────────────────────────────────────────────
port = PREFERRED_PORT
for attempt in range(8):
    if port_is_free(port):
        break
    print(f"[start] Port {port} busy, retry {attempt+1}/8...")
    time.sleep(1)
    free_port(port)
else:
    for alt in range(8081, 8090):
        if port_is_free(alt):
            port = alt
            print(f"[start] Port {PREFERRED_PORT} unavailable, using {port}")
            break

# ── Start ──────────────────────────────────────────────────────────
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print(f"[start] Starting on http://localhost:{port}")
import uvicorn
uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
