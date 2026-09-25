"""Keep tests from sending traces to Langfuse: env vars take precedence over .env."""

import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
