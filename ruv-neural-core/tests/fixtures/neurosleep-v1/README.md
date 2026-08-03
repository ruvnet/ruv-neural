# NeuroSleep V1 golden fixtures

`valid_bundle.json` is generated deterministically from the fixed test key
`[7; 32]` and payload in `attestation::tests::payload`. Its RFC 8785 payload
digest is `171eaaaa654b5a16de4605d603aa5d7c97db6784c624354b6ee97ca2ac9b83b7`.

`trust_profile.json` is deliberately separate from the bundle. It models key
enrollment and must never be interpreted as bundle-supplied trust. The secret
key is a public test fixture only and must not be used outside tests.
