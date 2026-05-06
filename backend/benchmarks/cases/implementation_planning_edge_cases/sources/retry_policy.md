# Webhook Retry Policy

Webhook delivery should retry transient 5xx and timeout failures. Use idempotency keys on every delivery attempt, exponential backoff with jitter, a dead-letter queue for exhausted retries, and tests for duplicate delivery, replay, and failed-event handling.
