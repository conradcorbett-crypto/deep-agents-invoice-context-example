"""Fake invoices and helpers for the context= Deep Agents example."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from typing_extensions import TypedDict


_PLACEHOLDER_OPENAI_KEY = "your_openai_api_key_here"


def find_demo_env_file() -> Path | None:
    """Return the demo ``.env`` path if it exists on disk."""
    demo_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = [
        demo_root,
        Path.cwd(),
        *Path.cwd().parents,
    ]
    seen: set[Path] = set()
    for base in candidates:
        resolved = base.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        env_path = resolved / ".env"
        if env_path.is_file():
            return env_path
    return None


def load_demo_env() -> Path | None:
    """Load a local ``.env`` so Jupyter kernels can see API keys.

    Cursor and VS Code notebook kernels do not inherit environment variables
    exported in a separate terminal. Place ``OPENAI_API_KEY`` in ``.env``
    (see ``.env.example``).

    Uses ``override=True`` so a blank ``OPENAI_API_KEY`` already present in the
    kernel environment does not block values from ``.env``.

    Returns:
        Path to the loaded ``.env`` file, or ``None`` if none was found.
    """
    from dotenv import load_dotenv

    env_path = find_demo_env_file()
    if env_path is not None:
        load_dotenv(env_path, override=True)
        return env_path
    load_dotenv(override=True)
    return None


def require_openai_api_key() -> None:
    """Raise if ``OPENAI_API_KEY`` is missing after loading ``.env``."""
    env_path = load_demo_env()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key and key != _PLACEHOLDER_OPENAI_KEY:
        return

    lines = [
        "OPENAI_API_KEY is not set in this Jupyter kernel.",
        "Terminal exports are not passed to the notebook process.",
        "Create .env in this project root (see .env.example),",
        "restart the kernel, then re-run the notebook cells.",
    ]
    if env_path is not None:
        lines.append(f"Found .env at {env_path} but OPENAI_API_KEY is empty or still the placeholder.")
    else:
        lines.append(f"Could not find .env (cwd={Path.cwd()}). Select the project .venv kernel.")
    msg = "\n".join(lines)
    raise OSError(msg)


class InvoiceContext(TypedDict):
    """Runtime context passed via ``context=`` at invoke time."""

    invoice_ids: list[str]


@dynamic_prompt
def invoice_scope_prompt(request: ModelRequest) -> str:
    """Inject scoped invoice IDs from invoke-time ``context=`` into the system prompt."""
    invoice_ids = request.runtime.context.get("invoice_ids", [])
    ids_text = ", ".join(invoice_ids) if invoice_ids else "(none)"
    return (
        "You analyze invoices using the analyze_invoice tool. "
        f"Scoped invoice IDs for this run (from runtime context, not the user message): "
        f"{ids_text}. "
        "Call analyze_invoice once per scoped ID, then summarize the results."
    )


@dataclass(frozen=True)
class InvoiceRecord:
    """Structured invoice payload stored outside the message list."""

    invoice_id: str
    vendor: str
    amount_usd: float
    status: str
    line_items: tuple[str, ...]


def build_fake_invoices() -> dict[str, InvoiceRecord]:
    """Return a small registry of fake invoices keyed by ID."""
    return {
        "INV-001": InvoiceRecord(
            invoice_id="INV-001",
            vendor="Acme Office Supplies",
            amount_usd=1248.50,
            status="pending",
            line_items=("Standing desks x4", "Monitor arms x4"),
        ),
        "INV-002": InvoiceRecord(
            invoice_id="INV-002",
            vendor="CloudScale Hosting",
            amount_usd=3890.00,
            status="approved",
            line_items=("GPU instances (March)", "Object storage"),
        ),
        "INV-003": InvoiceRecord(
            invoice_id="INV-003",
            vendor="Metro Catering Co.",
            amount_usd=612.75,
            status="flagged",
            line_items=("Team lunch", "Late delivery fee"),
        ),
    }


def format_invoice_analysis(invoice: InvoiceRecord) -> str:
    """Render a concise analysis string for tool output."""
    items = ", ".join(invoice.line_items)
    return (
        f"{invoice.invoice_id} | vendor={invoice.vendor} | "
        f"amount=${invoice.amount_usd:,.2f} | status={invoice.status} | items=[{items}]"
    )


def build_analyze_invoice_tool(
    registry: dict[str, InvoiceRecord],
) -> Any:
    """Build a tool that reads scoped invoice IDs from ``runtime.context``."""

    @tool
    def analyze_invoice(invoice_id: str, runtime: ToolRuntime[InvoiceContext]) -> str:
        """Analyze one invoice by ID using IDs supplied in runtime context."""
        allowed_ids = runtime.context["invoice_ids"]
        if invoice_id not in allowed_ids:
            return f"Invoice {invoice_id} is not in the current context scope: {allowed_ids}"
        invoice = registry.get(invoice_id)
        if invoice is None:
            return f"Invoice {invoice_id} was not found in the registry"
        return format_invoice_analysis(invoice)

    return analyze_invoice


def pretty_print_messages(result: dict[str, Any]) -> None:
    """Print assistant and tool messages from an agent invoke result."""
    for message in result["messages"]:
        role = message.__class__.__name__
        if role == "HumanMessage":
            continue
        if role == "AIMessage" and message.tool_calls:
            for call in message.tool_calls:
                print(f"AIMessage → tool_call {call['name']}({json.dumps(call['args'])})")
            continue
        text = getattr(message, "text", None) or str(message.content)
        print(f"{role}: {text}")
