"""Tkinter popup window that shows a reaction image/gif for the current expression state."""
import tkinter as tk

from PIL import Image, ImageSequence, ImageTk


class ReactionPopup:
    def __init__(self, image_map, on_close=None):
        self.image_map = image_map  # state name -> file path
        self.on_close = on_close
        self.current_state = None

        self.root = tk.Tk()
        self.root.title("Face Reaction")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)

        self.label = tk.Label(self.root, bg="black")
        self.label.pack()

        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 11)).pack(fill="x")

        self._frames = []
        self._durations = []
        self._frame_index = 0
        self._anim_job = None
        self._frame_cache = {}

    def set_status(self, text):
        self.status_var.set(text)

    def show_state(self, state):
        """Swap the displayed image, but only if the state actually changed."""
        if state == self.current_state:
            return
        self.current_state = state
        self.set_status(state.capitalize())

        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None

        path = self.image_map.get(state)
        if not path:
            return

        self._frames, self._durations = self._load_frames(path)
        self._frame_index = 0
        self._animate()

    def _load_frames(self, path):
        if path in self._frame_cache:
            return self._frame_cache[path]
        img = Image.open(path)
        frames, durations = [], []
        for frame in ImageSequence.Iterator(img):
            frames.append(ImageTk.PhotoImage(frame.convert("RGBA")))
            durations.append(frame.info.get("duration", 100) or 100)
        self._frame_cache[path] = (frames, durations)
        return frames, durations

    def _animate(self):
        if not self._frames:
            return
        self.label.configure(image=self._frames[self._frame_index])
        delay = self._durations[self._frame_index]
        self._frame_index = (self._frame_index + 1) % len(self._frames)
        self._anim_job = self.root.after(delay, self._animate)

    def after(self, delay_ms, callback):
        return self.root.after(delay_ms, callback)

    def mainloop(self):
        self.root.mainloop()

    def destroy(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _handle_close(self):
        if self.on_close:
            self.on_close()
        self.destroy()
