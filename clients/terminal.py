import logging
import uuid

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Input, Markdown, Static

from bus import Bus, Interrupt, Token, TurnDone, UsageReport, UserMessage

logger = logging.getLogger()

MOON_PHASES = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]


class ChatApp(App):
    STATE_STREAMING = "streaming"
    STATE_IDLE = "idle"
    # Overrides Textual's default ctrl+c -> help_quit system binding, which would
    # otherwise shadow the interrupt (key handlers lose to bindings on bubbling).
    BINDINGS = [Binding("ctrl+c", "interrupt_turn", show=False)]
    CSS = """
    #stats_bar {
        border: solid $accent;
        background: $surface;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, bus: Bus, context_size: int = 16384) -> None:
        super().__init__()
        self.bus = bus
        self.context_size = context_size or 16384

    async def on_mount(self):
        # This ensures the input is ready for typing immediately
        self.query_one(Input).focus()
        container = self.query_one("#chat_container")
        container.anchor()
        await container.mount(Static("[bold cyan]↩  C H A T[/]"))
        await container.mount(Static("[grey]Local AI Assistant[/]"))

        self.state = ChatApp.STATE_IDLE
        self.busy = False
        self.done = False
        self._frame_idx = 0
        self._pending: set[str] = set()
        self._stats_text = f"tokens: - / {self.context_size}"

        self.set_interval(0.2, self.tick_spinner)
        self.run_worker(write_output(self, self.bus))

    def compose(self) -> ComposeResult:
        # Use a scrollable container for the chat history
        yield VerticalScroll(id="chat_container")
        yield Static(f"tokens: - / {self.context_size}", id="stats_bar")
        yield Input(placeholder="Type here...")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip():
            return
        container = self.query_one("#chat_container")
        await container.mount(Static(f"[bold cyan]You:[/][white] {escape(event.value)}[/]"))
        container.scroll_end(animate=False)
        msg_id = uuid.uuid4().hex[:8]
        self.bus.post(UserMessage("0", event.value, msg_id))
        self._pending.add(msg_id)
        self.busy = True
        self.done = False
        self._frame_idx = 0
        self.render_bar()
        event.input.value = ""

    async def debug(self, text):
        container = self.query_one("#chat_container")
        await container.mount(Static(text))
        container.scroll_end(animate=False)

    async def create_container(self, markdown=False):
        container = self.query_one("#chat_container")
        widget = Markdown("") if markdown else Static("")
        await container.mount(widget)
        self.current_response = widget

    def action_interrupt_turn(self) -> None:
        self.bus.post(Interrupt("0"))

    def render_bar(self) -> None:
        if self.busy:
            frame = MOON_PHASES[self._frame_idx % len(MOON_PHASES)]
            self._frame_idx += 1
            prefix = f"{frame} "
        elif self.done:
            prefix = "✅ "
        else:
            prefix = "  "
        self.query_one("#stats_bar").update(f"{prefix}{self._stats_text}")

    def tick_spinner(self) -> None:
        if self.busy:
            self.render_bar()

    async def turn_done(self, msg_id: str) -> None:
        self._pending.discard(msg_id)
        if self._pending:
            return
        self.busy = False
        self.done = True
        self.render_bar()

    async def update_stats(self, usage: dict) -> None:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt_tokens + completion_tokens)
        ctx = self.context_size or 16384
        pct = (total / ctx * 100) if ctx else 0.0
        self._stats_text = (
            f"tokens: {total} / {ctx} ({pct:.1f}%) | prompt: {prompt_tokens} completion: {completion_tokens}"
        )
        self.render_bar()

    async def on_llm_text(self, user_id, role, token, chunk):
        """Now you can append tokens without a newline!"""
        if self.state == ChatApp.STATE_IDLE:
            if chunk:
                self.state = ChatApp.STATE_STREAMING
                self.current_text = ""
                if role == "thinking":
                    inner = Static("")
                    await self.query_one("#chat_container").mount(Collapsible(inner, title="💭 Thinking"))
                    self.current_response = inner
                    self.current_text = "[dim italic]"
                else:
                    await self.create_container(markdown=True)
            else:
                await self.create_container(markdown=False)

        if not chunk and token == "":
            # Per-LLM-call end marker (fires between tool calls too) - reset streaming state only.
            # The checkmark is set by the turn event, which fires once per user turn.
            self.state = ChatApp.STATE_IDLE
            return

        if role != "thinking" and chunk and getattr(self, "_thinking_active", False):
            self.current_response.update(self.current_text + "[/]")
            await self.create_container(markdown=True)
            self.current_text = ""
            self._thinking_active = False

        if role == "thinking" and chunk:
            self._thinking_active = True

        if self.state == ChatApp.STATE_STREAMING:
            self.current_text += escape(token)
            new_content = self.current_text
        else:
            new_content = f"[bold magenta]{role} 🤖:[/][white] {escape(token)}[/]"

        self.current_response.update(new_content)
        container = self.query_one("#chat_container")
        # Follow the stream only if the user is at the bottom (half-line
        # epsilon absorbs float noise from Textual's own pinning);
        # scroll_end also re-asserts the anchor, so follow survives any
        # interaction that released it (wheel, scrollbar, widget focus).
        if container.scroll_y + 0.5 >= container.max_scroll_y:
            container.scroll_end(animate=False)


async def write_output(app: ChatApp, bus: Bus):
    """Consume outbound events until the task is cancelled."""
    while True:
        event = await bus.outbound.get()
        try:
            match event:
                case Token(user_id, role, text, chunk):
                    await app.on_llm_text(user_id, role, text, chunk)
                case UsageReport(_, usage):
                    await app.update_stats(usage)
                case TurnDone(msg_id):
                    await app.turn_done(msg_id)
        except Exception:
            logger.exception("Failed to render outbound event: %s", event)


def stop() -> None:
    logger.info("Stopping terminal...")


async def run(bus: Bus, context_size: int = 0) -> None:
    """Run the Textual UI until cancelled."""
    await ChatApp(bus, context_size).run_async()
