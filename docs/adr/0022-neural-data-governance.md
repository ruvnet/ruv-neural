# ADR-0022 — Neural-data governance & neurorights compliance

## Status

Accepted — governance posture binding on the project; specific legal mappings
will be revised as law evolves (several instruments below are pending/soft-law).

## Context

Neurotechnology is now explicitly regulated, and the rules tightened sharply in
2024–2026. The project must position itself deliberately:

- **FDA software-device framework:** the **General Wellness** policy keeps
  *low-risk, non-disease* products out of device regulation — and the **2026
  revised guidance draws a hard line at invasiveness** (even minimally invasive
  is no longer low-risk) and tightens claims. For AI-enabled devices: the
  **PCCP final guidance** (Dec 2024) and the **AI lifecycle / GMLP** principles
  (2024–2025; lifecycle guidance still *draft*) govern any future SaMD path.
- **US state neural-data laws (four enacted):** Colorado **HB24-1058** (Aug
  2024), California **SB 1223** (Jan 2025; broadest — central *and* peripheral
  nervous system), Montana **SB 163** (Oct 2025; strongest consent regime),
  Connecticut **SB 1295** (CNS-only). A federal **MIND Act** is *proposed only*.
- **International:** Chile's constitutional neurorights reform (2021; standalone
  bill still pending); **UNESCO Recommendation on the Ethics of
  Neurotechnology** (adopted Nov 2025 — first global instrument, non-binding);
  **GDPR** Art. 9 special-category data; **EU AI Act** (emotion-recognition bans
  in workplace/education in force Feb 2025; biometric rules phasing to Aug 2026).

The project already has the right instincts: scope as wellness (ADR-0001),
consent-gated research flow (ADR-0014), tamper-evident audit (ADR-0009), and
data minimization (ADR-0021). This ADR makes governance an explicit, citable
posture rather than an emergent property.

## Decision

1. **Stay in the General Wellness / non-device lane, deliberately.** Non-invasive
   sensing + external sensory stimulation + **no diagnosis or treatment claim**
   (ADR-0001, ADR-0019, ADR-0020). Invasiveness and disease claims are the lines
   the 2026 FDA guidance hardens — the project sits firmly on the safe side.
2. **Treat neural data as sensitive/biometric by default**, regardless of which
   state law applies, satisfying the *strictest* enacted regime (CA SB 1223 /
   MT SB 163): explicit, revocable consent; purpose limitation; minimization and
   short retention (ADR-0021); and access/deletion support.
3. **Bake in transparency & auditability.** The hash-chained, optionally
   Ed25519-signed audit trail (ADR-0009) and the signed evidence export
   (ADR-0014) provide the record-keeping and user-facing transparency the GMLP
   transparency principles and neurorights laws call for.
4. **No emotion-recognition or covert profiling** in regulated contexts — aligned
   with the EU AI Act prohibitions; the project surfaces cognitive-state
   *topology*, not inferred emotions for workplace/education decisions.
5. **If a SaMD path is ever pursued**, it goes through the AI/ML framework
   (PCCP + GMLP + lifecycle management) as a *separate, gated* effort — never by
   silently relabeling the wellness platform.
6. **Maintain a living compliance map.** This ADR is revised (superseded) as the
   draft/pending instruments (FDA AI lifecycle guidance, Chile's bill, MIND Act,
   UNESCO follow-through, EU AI Act Aug-2026 provisions) finalize.

## Consequences

- The project has a defensible, documented answer to "is this regulated, and
  whose neural-data law applies?" — it complies with the strictest by default.
- Existing safety/consent/audit machinery is recognized as the compliance
  substrate, so governance reuses what's built rather than bolting on later.
- The wellness boundary is now defended on *three* fronts: science (ADR-0019),
  modality (ADR-0020), and law (this ADR).

## Evidence

- `docs/adr/0001-scope.md` — wellness-not-device scope (the General Wellness lane).
- `docs/adr/0009-audit-trail.md` — hash-chained, signable audit (transparency).
- `docs/adr/0014-web-console.md` — consent → contraindication → signed evidence
  export Research workflow.
- `docs/adr/0021-privacy-preserving-personalization.md` — minimization/retention
  controls this governance posture relies on.

## References

1. FDA revised General Wellness guidance (2026) — cov.com/en/news-and-insights/insights/2026/01/fda-issues-revised-guidance-on-general-wellness-products
2. FDA PCCP final guidance for AI-enabled devices (Dec 2024) — fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-software-medical-device
3. FDA Transparency for ML-Enabled Devices guiding principles (Jun 2024) — fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles
4. California SB 1223 & Colorado HB24-1058 (neural data) — afslaw.com/perspectives/alerts/california-and-colorado-establish-protections-neural-data
5. Montana SB 163 neural-data law — perkinscoie.com/insights/blog/dont-mind-if-i-do-montana-says-hands-neural-data
6. UNESCO Recommendation on the Ethics of Neurotechnology (Nov 2025) — unesco.org/en/legal-affairs/recommendation-ethics-neurotechnology
7. EU AI Act emotion-recognition prohibitions / biometrics & GDPR — iapp.org/news/a/biometrics-in-the-eu-navigating-the-gdpr-ai-act
8. Chile neurorights constitutional reform — courier.unesco.org/en/articles/chile-pioneering-protection-neurorights
