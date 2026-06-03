# Passing structured data to Deep Agents with `context=`

This example shows how to pass structured business data (invoice IDs and metadata) into a [Deep Agent](https://docs.langchain.com/oss/python/deepagents/overview) **without** embedding it in the user message.

The pattern uses **`context_schema`** at agent creation and **`context=`** at invoke time. Tools read scoped data from `runtime.context`. Context is immutable for the run and is **not** automatically injected into the model prompt — ideal for tenant IDs, scoped resource lists, credentials, and feature flags.

## What this demo does

1. Registers three fake invoices in an in-memory registry.
2. Creates a Deep Agent with `context_schema=InvoiceContext`.
3. Invokes the agent with a generic user message (`"Please analyze the attached invoices."`) and passes invoice IDs via `context={"invoice_ids": [...]}`.
4. The `analyze_invoice` tool reads allowed IDs from `runtime.context` and returns structured analysis for each invoice.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An OpenAI API key

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/deep-agents-invoice-context-example.git
cd deep-agents-invoice-context-example

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

./scripts/setup_kernel.sh
```

Open `invoice_context_demo.ipynb` in Cursor or VS Code and select the **Python (invoice-context-demo)** kernel. Run all cells.

### Manual setup

```bash
uv sync
uv run python -m ipykernel install --user --name invoice-context-demo --display-name "Python (invoice-context-demo)"
```

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
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=model,
    tools=[analyze_invoice],
    context_schema=InvoiceContext,  # TypedDict with invoice_ids: list[str]
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
