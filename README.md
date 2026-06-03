# Passing structured data to Deep Agents with `context=`

This example shows how to pass structured business data (invoice IDs and metadata) into a [Deep Agent](https://docs.langchain.com/oss/python/deepagents/overview) **without** embedding it in the user message.

The pattern uses **`context_schema`** at agent creation and **`context=`** at invoke time. Tools read scoped data from `runtime.context`. When the model must see scoped IDs (without putting them in the user message), **`@dynamic_prompt` middleware** reads the same runtime context at invoke time and builds the system prompt. Context is immutable for the run and is **not** automatically injected into the model prompt without middleware or tools.

## What this demo does

1. Registers three fake invoices in an in-memory registry.
2. Creates a Deep Agent with `context_schema=InvoiceContext`.
3. Invokes the agent with a generic user message (`"Please analyze the attached invoices."`) and passes invoice IDs via `context={"invoice_ids": [...]}`.
4. `invoice_scope_prompt` middleware injects scoped IDs from `context=` into the system prompt.
5. The `analyze_invoice` tool reads allowed IDs from `runtime.context` and returns structured analysis for each invoice.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An OpenAI API key

## Quick start

```bash
git clone https://github.com/conradcorbett-crypto/deep-agents-invoice-context-example.git
cd deep-agents-invoice-context-example

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

uv sync
```

Open `invoice_context_demo.ipynb` in Cursor or VS Code and select the project **`.venv`** Python interpreter. Run all cells.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for `gpt-4.1-mini` |
| `LANGSMITH_API_KEY` | No | Enable tracing in [LangSmith](https://smith.langchain.com/) |
| `LANGSMITH_TRACING` | No | Set to `true` to trace runs |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

**Note:** Jupyter kernels do not inherit environment variables exported in a terminal. Put keys in `.env` in this directory and restart the kernel after editing.

## Core pattern

```python
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from deepagents import create_deep_agent

@dynamic_prompt
def invoice_scope_prompt(request: ModelRequest) -> str:
    invoice_ids = request.runtime.context.get("invoice_ids", [])
    ids_text = ", ".join(invoice_ids) if invoice_ids else "(none)"
    return (
        "You analyze invoices using the analyze_invoice tool. "
        f"Scoped invoice IDs for this run: {ids_text}. "
        "Call analyze_invoice once per scoped ID, then summarize the results."
    )

agent = create_deep_agent(
    model=model,
    tools=[analyze_invoice],
    context_schema=InvoiceContext,  # TypedDict with invoice_ids: list[str]
    middleware=[invoice_scope_prompt],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Please analyze the attached invoices."}]},
    config={"configurable": {"thread_id": "my-thread"}},
    context={"invoice_ids": ["INV-001", "INV-002", "INV-003"]},
)
```

Inside the tool:

```python
@tool
def analyze_invoice(invoice_id: str, runtime: ToolRuntime[InvoiceContext]) -> str:
    allowed_ids = runtime.context["invoice_ids"]
    ...
```

## When to use `context=` vs other options

| Mechanism | Use for |
|-----------|---------|
| **`context=`** | Immutable per-run metadata: tenant IDs, scoped resource IDs, API keys, feature flags |
| **`state_schema` + invoke keys** | Data that should checkpoint with the thread or change during the conversation |
| **`config`** | Thread IDs and tracing only — not business payload |
| **`files=`** | Binary attachments (PDFs, images) via StateBackend or FilesystemBackend |

## Documentation

- [Context engineering in Deep Agents](https://docs.langchain.com/oss/python/deepagents/context-engineering)
- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview)

## License

MIT
