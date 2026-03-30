# Changelog
All notable changes to this project will be documented in this file.

## v1.1.0

### Added
- **Device Assurance Policy tools** — 5 new MCP tools to manage Device Assurance Policies across all platforms (Android, iOS, macOS, Windows, ChromeOS):
  - `list_device_assurance_policies` — list all policies with per-attribute security status
  - `get_device_assurance_policy` — retrieve a policy by ID
  - `create_device_assurance_policy` — create a policy with OS version validation
  - `replace_device_assurance_policy` — fully replace a policy with before/after diff and security implications
  - `delete_device_assurance_policy` — delete a policy with elicitation-guarded confirmation

### Changed
- Upgraded Okta Python SDK dependency to `v3.3.0`
- Updated all tool files (`applications`, `groups`, `policies`, `users`, `system_logs`) for SDK v3.3.0 conventions

### Fixed
- `delete_user` rename from `deactivate_or_delete_user` (SDK v3.x breaking change)
- `OktaSignOnPolicyRule` typed model usage so `SIGN_ON` policy rule actions are no longer silently dropped
- `GroupProfile` attribute error during group creation logging
- Stale cache and intermittent list failures in pagination

## v1.0.0

- Initial release of the self hosted okta-mcp-server.
