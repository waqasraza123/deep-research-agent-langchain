# Retry Guide B

Webhook retries should use idempotency keys, exponential backoff, and a dead-letter queue for repeated delivery failures.
