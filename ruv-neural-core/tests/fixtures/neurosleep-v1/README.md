# NeuroSleep V1 golden fixtures

`valid_bundle.json` is generated deterministically from the fixed test key
`[7; 32]` and payload in `attestation::tests::payload`. Its RFC 8785 payload
digest is `707230a26db16a58bbb9b8184936e98e3c9c1da89e054db29bf7c8b7cc2bd495`.

`trust_profile.json` is deliberately separate from the bundle. It models key
enrollment and must never be interpreted as bundle-supplied trust. The secret
key is a public test fixture only and must not be used outside tests.
