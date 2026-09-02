"""
Refactored BookTranslator GUI Application with thread-safe UIEventQueue,
AsyncTaskManager, ThrottledLogBuffer, and cooperative job cancellation.
Part of Milestone 1 (UI & Concurrency Optimization) for BookTranslator.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Optional

import customtkinter as ctk

from src.config.settings import settings
from src.launcher.concurrency import (
    AsyncTaskManager,
    CancellationToken,
    ProgressEvent,
    TaskState,
    UIEventQueue,
)
from src.launcher.throttled_logger import ThrottledLogBuffer
from src.launcher.translation_runner import execute_translation_job, count_existing_chunks

logger = logging.getLogger(__name__)


class GUIApp(ctk.CTk):
    """
    Main CustomTkinter application window for BookTranslator.
    All cross-thread communication is strictly marshaled through UIEventQueue.
    """

    def __init__(self) -> None:
        super().__init__()

        self.title("BookTranslator - AI Translation System")
        self.geometry("780x680")
        self.minsize(650, 550)
        self.grid_columnconfigure(1, weight=1)

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Concurrency & Event Subsystems
        self.ui_queue = UIEventQueue(poll_interval_ms=20, max_events_per_tick=50)
        self.ui_queue.start_polling(self)
        self.task_manager = AsyncTaskManager(self.ui_queue)

        # Build UI Sections
        self._init_file_section()
        self._init_model_section()
        self._init_progress_section()
        self._init_actions_section()
        self._init_console_section()

        # Window Close Protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_file_section(self) -> None:
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="ew")
        self.file_frame.grid_columnconfigure(1, weight=1)

        self.lbl_input = ctk.CTkLabel(self.file_frame, text="Вхідний файл:")
        self.lbl_input.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.input_var = ctk.StringVar()
        self.entry_input = ctk.CTkEntry(self.file_frame, textvariable=self.input_var, state="disabled")
        self.entry_input.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.btn_input = ctk.CTkButton(self.file_frame, text="Огляд...", width=80, command=self.browse_input)
        self.btn_input.grid(row=0, column=2, padx=10, pady=10)

        self.lbl_output = ctk.CTkLabel(self.file_frame, text="Вихідний файл:")
        self.lbl_output.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.output_var = ctk.StringVar()
        self.entry_output = ctk.CTkEntry(self.file_frame, textvariable=self.output_var, state="disabled")
        self.entry_output.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.btn_output = ctk.CTkButton(self.file_frame, text="Огляд...", width=80, command=self.browse_output)
        self.btn_output.grid(row=1, column=2, padx=10, pady=10)

    def _init_model_section(self) -> None:
        self.model_frame = ctk.CTkFrame(self)
        self.model_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.model_frame.grid_columnconfigure(1, weight=1)

        self.lbl_nllb = ctk.CTkLabel(self.model_frame, text="Версія NLLB (Етап 1):")
        self.lbl_nllb.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.nllb_models = [
            "facebook/nllb-200-distilled-600M",
            "facebook/nllb-200-distilled-1.3B",
            "facebook/nllb-200-3.3B",
        ]
        self.nllb_var = ctk.StringVar(value=settings.nllb_model_name)
        self.combo_nllb = ctk.CTkComboBox(self.model_frame, values=self.nllb_models, variable=self.nllb_var)
        self.combo_nllb.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.lbl_llm = ctk.CTkLabel(self.model_frame, text="Модель LLM (Етап 2):")
        self.lbl_llm.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.llm_models = [
            "CohereLabs/tiny-aya-global",
            "CohereForAI/aya-23-8B",
            "CohereForAI/aya-expanse-8b",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ]
        self.llm_var = ctk.StringVar(value=settings.aya_model_name)
        self.combo_llm = ctk.CTkComboBox(self.model_frame, values=self.llm_models, variable=self.llm_var)
        self.combo_llm.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.lbl_token = ctk.CTkLabel(self.model_frame, text="Hugging Face Token:")
        self.lbl_token.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.token_var = ctk.StringVar(value=settings.hf_token if settings.hf_token else "")
        self.entry_token = ctk.CTkEntry(self.model_frame, textvariable=self.token_var, show="*")
        self.entry_token.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        self.offline_var = ctk.BooleanVar(value=settings.offline_mode)
        self.chk_offline = ctk.CTkCheckBox(
            self.model_frame,
            text="Автономний режим (Використовувати лише завантажені моделі)",
            variable=self.offline_var,
        )
        self.chk_offline.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="w")

    def _init_progress_section(self) -> None:
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=(5, 5), sticky="ew")
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Статус: Очікування запуску", anchor="w")
        self.progress_label.grid(row=0, column=0, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.grid(row=1, column=0, pady=5, sticky="ew")
        self.progress_bar.set(0.0)

    def _init_actions_section(self) -> None:
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.action_frame.grid_columnconfigure(0, weight=1)
        self.action_frame.grid_columnconfigure(1, weight=1)
        self.action_frame.grid_columnconfigure(2, weight=1)

        self.btn_prompt = ctk.CTkButton(
            self.action_frame, text="📝 Редагувати Промпт", fg_color="gray", command=self.open_prompt_file
        )
        self.btn_prompt.grid(row=0, column=0, padx=5, pady=10, sticky="ew")

        self.btn_start = ctk.CTkButton(
            self.action_frame, text="▶ Почати Переклад", fg_color="green", command=self.on_start_clicked
        )
        self.btn_start.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        self.btn_cancel = ctk.CTkButton(
            self.action_frame,
            text="⏹ Зупинити",
            fg_color="darkred",
            state="disabled",
            command=self.on_cancel_clicked,
        )
        self.btn_cancel.grid(row=0, column=2, padx=5, pady=10, sticky="ew")

    def _init_console_section(self) -> None:
        self.console_textbox = ctk.CTkTextbox(self, wrap="word", height=200, state="disabled")
        self.console_textbox.grid(row=4, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="nsew")
        self.grid_rowconfigure(4, weight=1)

        # Initialize Throttled Log Buffer (50ms flusher, 1,000 max lines)
        self.log_buffer = ThrottledLogBuffer(self.console_textbox, flush_interval_ms=50, max_lines=1000)

    # --- UI Event Handlers ---

    def browse_input(self) -> None:
        filename = filedialog.askopenfilename(
            title="Оберіть вхідний файл",
            filetypes=[("Підтримувані формати", "*.txt;*.epub;*.fb2;*.docx;*.pdf")],
        )
        if filename:
            self.input_var.set(filename)
            out_path = Path(filename).with_name(Path(filename).stem + "_ukr" + Path(filename).suffix)
            self.output_var.set(str(out_path))

    def browse_output(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Зберегти як...",
            filetypes=[("Підтримувані формати", "*.txt;*.epub;*.fb2;*.docx;*.pdf")],
        )
        if filename:
            self.output_var.set(filename)

    def open_prompt_file(self) -> None:
        prompt_path = settings.prompt_file_path
        if prompt_path.exists():
            if platform.system() == "Windows":
                os.startfile(prompt_path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(prompt_path)])
            else:
                subprocess.Popen(["xdg-open", str(prompt_path)])
        else:
            messagebox.showerror("Помилка", f"Файл промпту не знайдено:\n{prompt_path}")

    def save_settings(self) -> None:
        settings.save_to_env("NLLB_MODEL_NAME", self.nllb_var.get())
        settings.save_to_env("AYA_MODEL_NAME", self.llm_var.get())
        settings.save_to_env("OFFLINE_MODE", str(self.offline_var.get()))
        if self.token_var.get().strip():
            settings.save_to_env("HF_TOKEN", self.token_var.get().strip())

        settings.nllb_model_name = self.nllb_var.get()
        settings.aya_model_name = self.llm_var.get()
        settings.offline_mode = self.offline_var.get()
        settings.hf_token = self.token_var.get().strip()

    def on_start_clicked(self) -> None:
        input_path = self.input_var.get()
        output_path = self.output_var.get()

        if not input_path or not output_path:
            messagebox.showwarning("Увага", "Будь ласка, оберіть вхідний та вихідний файли.")
            return

        if not Path(input_path).exists():
            messagebox.showerror("Помилка", f"Вхідний файл не існує:\n{input_path}")
            return

        self.save_settings()

        # Check for existing cache and prompt user
        existing_chunks = count_existing_chunks(input_path)
        clear_cache_flag = False
        if existing_chunks > 0:
            result = messagebox.askyesnocancel(
                "Відновлення перекладу",
                f"Знайдено перерваний або незавершений переклад цього файлу ({existing_chunks} збережених фрагментів).\n\n"
                "Бажаєте продовжити переклад з місця зупинки?\n"
                "Так - продовжити\nНі - почати заново (кеш буде видалено)\nСкасувати - відміна"
            )
            if result is None:
                return
            elif result is False:
                clear_cache_flag = True

        # Update UI Controls to active running state
        self.btn_start.configure(state="disabled", text="Переклад триває...")
        self.btn_cancel.configure(state="normal", text="⏹ Зупинити")
        self.btn_input.configure(state="disabled")
        self.btn_output.configure(state="disabled")
        self.combo_nllb.configure(state="disabled")
        self.combo_llm.configure(state="disabled")
        self.progress_bar.set(0.0)
        self.progress_label.configure(text="Ініціалізація перекладу...")

        # Launch worker via AsyncTaskManager (100% thread-safe)
        self.task_manager.submit_task(
            task_func=lambda cancel_tok, prog_cb: execute_translation_job(
                input_file=Path(input_path),
                output_file=Path(output_path),
                cancellation_token=cancel_tok,
                progress_callback=prog_cb,
                clear_cache=clear_cache_flag,
            ),
            on_progress=self._handle_task_progress,
            on_success=self._handle_task_success,
            on_error=self._handle_task_error,
            on_cancelled=self._handle_task_cancelled,
        )

    def on_cancel_clicked(self) -> None:
        self.btn_cancel.configure(state="disabled", text="Зупинка...")
        self.progress_label.configure(text="Зупинка процесу перекладу... Будь ласка, зачекайте.")
        self.task_manager.cancel_task()

    # --- UI Marshaled Callbacks (Executed strictly on Main Tkinter Thread) ---

    def _handle_task_progress(self, event: ProgressEvent) -> None:
        self.progress_bar.set(event.percentage)
        mins, secs = divmod(int(event.eta_seconds), 60)
        hours, mins = divmod(mins, 60)

        if hours > 0:
            eta_str = f"{hours} год {mins} хв"
        elif mins > 0:
            eta_str = f"{mins} хв {secs} сек"
        else:
            eta_str = f"{secs} сек"

        pct_int = int(event.percentage * 100)
        if event.total > 0:
            self.progress_label.configure(
                text=f"[{event.stage_name}] Чанк {event.current}/{event.total} ({pct_int}%). Залишилось: ~{eta_str}"
            )
        else:
            msg = event.status_message or event.message or event.stage_name
            self.progress_label.configure(text=f"[{event.stage_name}] {msg}")

    def _handle_task_success(self, result: Any) -> None:
        self._reset_controls_to_idle()
        self.progress_bar.set(1.0)
        self.progress_label.configure(text="Переклад успішно завершено!")
        messagebox.showinfo("Готово", f"Переклад успішно завершено!\nЗбережено у:\n{self.output_var.get()}")

    def _handle_task_error(self, ex: Exception, tb_str: str) -> None:
        self._reset_controls_to_idle()
        self.progress_label.configure(text=f"Помилка: {str(ex)}")
        messagebox.showerror("Помилка", f"Сталася помилка під час перекладу (деталі у лозі):\n{str(ex)}")

    def _handle_task_cancelled(self) -> None:
        self._reset_controls_to_idle()
        self.progress_label.configure(text="Переклад було зупинено користувачем.")
        messagebox.showinfo("Зупинено", "Процес перекладу успішно зупинено. Прогрес збережено у базі.")

    def _reset_controls_to_idle(self) -> None:
        self.btn_start.configure(state="normal", text="▶ Почати Переклад")
        self.btn_cancel.configure(state="disabled", text="⏹ Зупинити")
        self.btn_input.configure(state="normal")
        self.btn_output.configure(state="normal")
        self.combo_nllb.configure(state="normal")
        self.combo_llm.configure(state="normal")

    def on_closing(self) -> None:
        """Graceful window teardown protocol."""
        if self.task_manager.is_running():
            if messagebox.askyesno("Підтвердження", "Переклад триває. Ви дійсно бажаєте зупинити процес та вийти?"):
                self.withdraw()  # Instantly hide window to give immediate visual feedback
                self.task_manager.cancel_task()
                self.task_manager.join_worker(timeout=2.0)
            else:
                return

        # Cleanup logging & UI event polling before destroying widget
        self.log_buffer.shutdown()
        self.ui_queue.stop_polling()
        self.destroy()


if __name__ == "__main__":
    app = GUIApp()
    app.mainloop()
