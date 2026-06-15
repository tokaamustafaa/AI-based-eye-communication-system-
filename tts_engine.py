"""
tts_engine.py — Text-to-speech via Windows SAPI (PowerShell).

Uses a dedicated queue + thread so the gaze loop is never blocked.
PowerShell SAPI is used directly — no pyttsx3 COM threading issues.
"""

import queue
import threading
import subprocess

_q: "queue.Queue[tuple | None]" = queue.Queue()
_worker_thread: threading.Thread | None = None


def _worker() -> None:
    while True:
        item = _q.get()
        if item is None:
            return
        text, done_event = item
        safe = text.replace("'", "''")   # escape single quotes for PS
        cmd = (
            f"$v = New-Object -ComObject SAPI.SpVoice; "
            f"$v.Rate = 0; "
            f"$v.Speak('{safe}')"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                capture_output=True,
                timeout=15,
            )
        except Exception as exc:
            print(f"[TTS] error: {exc}")
        finally:
            if done_event is not None:
                done_event.set()


def _ensure_worker() -> None:
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker, daemon=True)
        _worker_thread.start()


def speak(text: str, block: bool = False) -> None:
    """
    Speak `text` aloud.

    Args:
        text  : word or phrase to say
        block : True → wait until speech finishes
                False → fire-and-forget
    """
    _ensure_worker()
    if block:
        done = threading.Event()
        _q.put((text, done))
        done.wait()
    else:
        _q.put((text, None))


def list_voices() -> list:
    return []


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Speaking test phrase …")
    speak("Gaze communication system is ready.", block=True)
    print("Done.")
