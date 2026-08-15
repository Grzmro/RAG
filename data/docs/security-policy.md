# Information Security Policy

Northwind Analytics Ltd. Policy ID SEC-001, revision 7, approved 20 January 2025 by
the Security Steering Group.

## Scope

This policy covers all systems that process customer data, all employee endpoints,
and all third-party services holding Northwind or customer data.

## Access control

Access is granted on a least-privilege basis and reviewed quarterly. Multi-factor
authentication is mandatory for every system that handles customer data; hardware
security keys are required for production infrastructure access.

Production database credentials are issued as short-lived tokens valid for **8 hours**
and are never stored in source control. Standing production access is limited to the
on-call rota and expires automatically when the rota shift ends.

## Data classification

Four levels apply:

1. **Public** — may be shared freely.
2. **Internal** — default for company documents; not for external distribution.
3. **Confidential** — customer data, contracts, financials; encrypted at rest and in
   transit, access logged.
4. **Restricted** — credentials, encryption keys, personal data of end users; access
   requires named approval from the Security Steering Group.

Customer telemetry ingested through Atlas is classified **Confidential** by default and
is retained for 24 months unless the customer contract specifies otherwise.

## Incident response

Suspected incidents must be reported to `security@northwind.example` within **one hour**
of discovery. The on-call security engineer triages within 30 minutes and assigns a
severity from SEV-1 (customer data exposure or full outage) to SEV-4 (no customer
impact).

SEV-1 and SEV-2 incidents require a written post-incident review published internally
within **5 working days**. Regulatory notification for personal-data breaches is made
within 72 hours of confirmation, in line with UK GDPR.

## Vendor review

Any third-party service processing Confidential or Restricted data must pass a
security review before procurement. Reviews are repeated annually and cover SOC 2
Type II or ISO 27001 evidence, sub-processor lists, and data residency.

## Employee obligations

All employees complete security training within 14 days of joining and annually
thereafter. Devices must use full-disk encryption and lock after 5 minutes idle.
Reporting a suspected phishing attempt is never penalised, even if the employee
interacted with the message.
