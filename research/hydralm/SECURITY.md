# Security Policy

## Supported versions

HydraLM is a research project. Only the most recent minor release line
receives security fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |
| < 0.1   | no        |

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security problems.

Email [security@byte271.dev](mailto:security@byte271.dev) with:

- A description of the issue and its impact.
- Steps to reproduce, including a minimal code sample.
- Any relevant logs or stack traces.
- Your name and affiliation if you would like to be credited.

You can expect:

- An acknowledgement within 3 business days.
- A triage assessment within 10 business days.
- A fix or mitigation plan within 30 days for confirmed issues, faster
  for anything actively exploitable.

We publish advisories through GitHub Security Advisories on
https://github.com/byte271/hydralm/security/advisories once a fix is
released.

## Scope

Reports covering the following are in scope:

- Arbitrary-code-execution or memory-safety issues triggered by model
  loading, checkpoint parsing, or the `HFCompatibleAdapter`.
- Deserialisation of attacker-controlled checkpoints leading to
  privilege escalation beyond what plain `torch.load` already implies.
- Denial-of-service issues in the streaming decoder or fact bank that
  persist after a session is closed.

Out of scope:

- Pure model-quality issues (hallucinations, bias, prompt injection at
  the application layer).
- Issues requiring a compromised Python environment.
- Issues that only affect unsupported versions.

## Safe harbour

We will not pursue legal action against researchers who:

- Make a good-faith effort to avoid privacy violations, data
  destruction, and service disruption while researching.
- Only interact with accounts they own or with explicit permission.
- Give us a reasonable opportunity to fix the issue before public
  disclosure.
