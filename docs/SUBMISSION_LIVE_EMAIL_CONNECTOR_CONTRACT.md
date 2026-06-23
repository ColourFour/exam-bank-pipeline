# Submission Live Email Connector Contract

This contract defines Phase 5B: a read-only live email connector layer for assignment submissions.

## Purpose

The live connector is a transport adapter only. It reads a configured assignment-specific mailbox scope and converts messages into the same `InboundEmailMessage` and `InboundEmailAttachment` models used by fixture-backed Phase 4 intake. Downstream intake, quarantine, Phase 1 validation, outgoing approval, and draft grading rules must not care whether the source was a fixture, local export, or live connector.

## Safety Defaults

- Read-only is the default.
- Dry-run is the default.
- Full-inbox scans are forbidden by default.
- A run must specify an assignment label, folder, or search query.
- Attachments are not staged and Phase 1 is not called in dry-run mode.
- Apply mode requires an explicit `--apply` flag.
- No live sending is performed by this connector.
- Student-facing output still requires the Phase 5 outgoing approval gate.

## Source Scope

The connector may inspect only the configured `mailbox_scope` and/or `search_query` for one assignment. The first real run should use a dedicated label or folder plus a narrow subject/attachment query. Broad mailbox access must be treated as unsafe until a future contract expands the evidence base.

## Credential Privacy

Credentials, auth files, cookies, refresh data, and tokens must never be committed. Real connector config belongs under ignored private roots such as:

```text
data/submissions/<assignment_id>/email_connector_config.json
```

Templates may be committed only when they contain placeholders and no secrets.

## Dry-Run Requirements

Dry-run may list scoped messages, fetch safe metadata, evaluate student matching, evaluate attachment status, and write private readiness artifacts. Dry-run must not stage attachments, call Phase 1, queue outgoing mail, send mail, or write credentials.

## Apply Requirements

Apply mode may materialize scoped messages into private fixture-compatible artifacts, call existing Phase 4 intake, run Phase 1 validation through that handoff, preserve provenance, and write audit logs. Apply mode still must not send outgoing email.

## Audit And Evidence

Runs write local audit and readiness artifacts under ignored submission roots. The first-assignment readiness report must include assignment ID, roster count, mailbox scope/query, messages found, likely accepted/quarantined counts, apply recommendation, and warnings.

## Fixture Parity

Fixture-backed intake remains the test source. Fake connectors and local exports must prove that connector messages become the same internal models and follow the same quarantine and validation path as Phase 4 fixtures.

## Rollback And Disable

Disable a connector by removing or renaming the private connector config, removing the assignment label/folder scope, or running only dry-run. Because apply mode writes only ignored private artifacts, rollback is local cleanup under `output/submissions/<assignment_id>/` and `reports/submissions/`.

## Provider Status

Phase 5B implements the provider-neutral interface, fake connector, and local-export connector. Direct mailbox providers are deferred until credential handling and first-run evidence justify adding them.
