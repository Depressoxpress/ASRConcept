"""
overlay.py — Transparent, Always-On-Top Tkinter Overlay

Run as a standalone process.  Reads new transcript lines from a temp file
that the main Streamlit app writes to.  Displays scrolling text over all
other windows.

Usage (launched automatically by main.py):
    python overlay.py <transcript_file> <control_file>
"""

import sys
import os
import tkinter as tk
import time
import threading


class TranscriptOverlay:
    """Semi-transparent always-on-top overlay showing live transcript text."""

    # ── Appearance defaults ──────────────────────────────────────────
    BG_COLOR = "#1a1a2e"
    FG_COLOR = "#e0e0e0"
    FONT_FAMILY = "Consolas"
    FONT_SIZE = 13
    DEFAULT_ALPHA = 0.82
    WIDTH_CHARS = 60
    HEIGHT_LINES = 8
    PADDING = 12

    def __init__(
        self,
        transcript_file: str,
        control_file: str,
        opacity: float = DEFAULT_ALPHA,
    ):
        self._transcript_file = transcript_file
        self._control_file = control_file
        self._last_read_pos = 0

        # ── Root window ──────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("ASR Overlay")
        self.root.configure(bg=self.BG_COLOR)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", opacity)
        self.root.overrideredirect(True)  # remove title bar

        # Position: bottom-right of screen
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w, win_h = 620, 230
        x = screen_w - win_w - 30
        y = screen_h - win_h - 60
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # ── Dragging support ─────────────────────────────────────────
        self._drag_data = {"x": 0, "y": 0}
        self.root.bind("<ButtonPress-1>", self._on_drag_start)
        self.root.bind("<B1-Motion>", self._on_drag_motion)

        # ── Header bar ───────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#16213e", height=28)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🎙 ASR Overlay",
            bg="#16213e",
            fg="#8899aa",
            font=(self.FONT_FAMILY, 10, "bold"),
        ).pack(side=tk.LEFT, padx=8)

        close_btn = tk.Label(
            header,
            text=" ✕ ",
            bg="#16213e",
            fg="#cc5555",
            font=(self.FONT_FAMILY, 11, "bold"),
            cursor="hand2",
        )
        close_btn.pack(side=tk.RIGHT, padx=4)
        close_btn.bind("<Button-1>", lambda e: self.root.destroy())

        # ── Text display ─────────────────────────────────────────────
        self.text_widget = tk.Text(
            self.root,
            wrap=tk.WORD,
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
            font=(self.FONT_FAMILY, self.FONT_SIZE),
            bd=0,
            padx=self.PADDING,
            pady=self.PADDING,
            highlightthickness=0,
            insertbackground=self.BG_COLOR,  # hide cursor
            state=tk.DISABLED,
            spacing3=4,
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        # ── Polling loop ─────────────────────────────────────────────
        self._poll()

    # ── Drag handlers ────────────────────────────────────────────────
    def _on_drag_start(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag_motion(self, event):
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")

    # ── File polling ─────────────────────────────────────────────────
    def _poll(self):
        """Check the transcript file and control file every 150 ms."""

        # Check control file for commands
        try:
            if os.path.exists(self._control_file):
                with open(self._control_file, "r", encoding="utf-8") as f:
                    cmd = f.read().strip()
                if cmd == "CLEAR":
                    self._clear_text()
                    self._last_read_pos = 0
                    # Acknowledge
                    with open(self._control_file, "w", encoding="utf-8") as f:
                        f.write("")
                elif cmd == "EXIT":
                    self.root.destroy()
                    return
        except Exception:
            pass

        # Read new transcript data
        try:
            if os.path.exists(self._transcript_file):
                with open(self._transcript_file, "r", encoding="utf-8") as f:
                    content = f.read()
                if len(content) > self._last_read_pos:
                    new_text = content[self._last_read_pos:]
                    self._last_read_pos = len(content)
                    self._append_text(new_text)
        except Exception:
            pass

        self.root.after(150, self._poll)

    # ── Text helpers ─────────────────────────────────────────────────
    def _append_text(self, text: str):
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state=tk.DISABLED)

    def _clear_text(self):
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.configure(state=tk.DISABLED)

    # ── Run ──────────────────────────────────────────────────────────
    def run(self):
        self.root.mainloop()


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python overlay.py <transcript_file> <control_file>")
        sys.exit(1)

    transcript_path = sys.argv[1]
    control_path = sys.argv[2]
    opacity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.82

    overlay = TranscriptOverlay(transcript_path, control_path, opacity)
    overlay.run()
