# RAG evaluation report

Generated: 2026-08-15T14:38:23+00:00
Answer model: `claude-haiku-4-5-20251001` · Judge model: `claude-sonnet-5` · top_k=5 · chunk=900/150 · embeddings=`fastembed:BAAI/bge-small-en-v1.5` · rerank=`none` · retrieval=`hybrid` · citations=`markers`

## Summary

| Metric | Value |
| --- | --- |
| questions | 20 |
| mean groundedness | 4.95 |
| grounded rate pct | 100.0 |
| hallucination rate pct | 0.0 |
| flagged rate pct | 0.0 |
| citation validity pct | 100.0 |
| citation span integrity pct | 0.0 |
| dangling citation count | 0 |
| abstention accuracy pct | 100.0 |
| retrieval recall pct | 100.0 |
| must contain pass pct | 100.0 |

## Per-question results

| ID | Groundedness | Verdict | Cited sources | Flags |
| --- | --- | --- | --- | --- |
| lex-atlas-key-header | 5 | grounded | atlas-product-spec.md | — |
| lex-dlq-topic-name | 5 | grounded | atlas-product-spec.md | — |
| lex-policy-id | 5 | grounded | security-policy.md | — |
| lex-v0-removal | 5 | grounded | atlas-product-spec.md | — |
| lex-backoff-window | 5 | grounded | atlas-product-spec.md | — |
| lex-remote-waiver-distance | 5 | grounded | employee-handbook.md | — |
| trap-training-deadline | 5 | grounded | security-policy.md | — |
| trap-device-lock | 5 | grounded | security-policy.md | — |
| trap-rate-limit-increase | 5 | grounded | atlas-product-spec.md | — |
| trap-review-cadences | 5 | grounded | security-policy.md | — |
| trap-who-approves | 5 | grounded | security-policy.md, employee-handbook.md | — |
| cross-atlas-key-classification | 5 | grounded | security-policy.md | — |
| cross-vendor-atlas-telemetry | 4 | grounded | security-policy.md | — |
| cross-escalation-paths | 5 | grounded | atlas-product-spec.md, security-policy.md | — |
| unans-retry-after-value | 5 | grounded | atlas-product-spec.md | — |
| unans-dlq-replicas | 5 | grounded | atlas-product-spec.md | — |
| unans-p99-latency | 5 | grounded | atlas-product-spec.md | — |
| unans-encryption-algorithm | 5 | grounded | security-policy.md | — |
| unans-sick-note-validity | 5 | grounded | — | — |
| unans-early-laptop-refresh | 5 | grounded | — | — |
