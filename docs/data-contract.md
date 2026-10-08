# Data Contract

This document defines the common event format shared between the Big Data pipeline and the AI pipeline.

## Event Schema

| Field | Type | Description |
|---|---|---|
| event_id | string | Unique event identifier |
| timestamp | datetime | Time at which the event occurred |
| user | string | User or account associated with the event |
| src_ip | string | Source IP address |
| dst_host | string | Destination host |
| event_type | string | Type of security/system event |
| status | string | Event status |
| severity | integer | Event severity |
| incident_id | string | Correlated incident identifier |

## Example Event

```text
event_id: EVT001
timestamp: 2026-10-08 10:01:00
user: admin
src_ip: 192.168.1.10
dst_host: server01
event_type: LOGIN_FAILURE
status: FAILED
severity: 1
incident_id: INC001