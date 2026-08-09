# Atlas Ingest Service — Technical Specification

Document owner: Platform Team. Status: approved. Last revised: 12 February 2025.

## Purpose

Atlas is Northwind Analytics' batch and streaming ingestion service. It accepts
customer telemetry, validates it against a registered schema, and writes conforming
records to the analytics lake.

## Interfaces

Atlas exposes two entry points:

- `POST /v1/batches` — accepts newline-delimited JSON, maximum payload **50 MB** per
  request, maximum 100,000 records per batch.
- A Kafka topic `atlas.ingest.v1` for streaming producers, partitioned by tenant ID.

Both interfaces require an API key passed in the `X-Atlas-Key` header. Keys are
scoped to a single tenant and are rotated automatically every 90 days.

## Validation and error handling

Every record is validated against the tenant's registered Avro schema. Records that
fail validation are written to the dead-letter topic `atlas.dlq.v1` together with a
`rejection_reason` field. Records are retained in the dead-letter topic for **14 days**.

Atlas retries transient downstream failures up to **5 times** with exponential backoff
starting at 250 ms and capped at 30 seconds. After the fifth failure the batch is
marked `failed` and an alert is raised to the on-call Platform engineer.

## Service level objectives

- Availability: 99.9% monthly for the batch endpoint, 99.5% for the streaming path.
- Ingestion latency: p95 under 90 seconds from acceptance to lake availability.
- Data durability: three replicas across two availability zones.

Breaching the availability SLO in two consecutive months triggers a formal incident
review chaired by the Head of Platform.

## Rate limits

Each tenant is limited to **60 batch requests per minute** and 20,000 streaming
records per second. Requests over the limit receive HTTP 429 with a `Retry-After`
header. Limit increases are requested through the customer's account manager and
take effect within two business days.

## Deprecation policy

Breaking changes to the `/v1` API require 180 days of notice. Non-breaking additive
changes may ship without notice. The `/v0` endpoints were removed on 1 January 2025.
