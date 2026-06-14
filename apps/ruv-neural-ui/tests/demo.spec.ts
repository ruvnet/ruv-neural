import { test, expect } from "@playwright/test";

/**
 * End-to-end acceptance (ADR-0014 §16): a visitor can run a deterministic Ruflo
 * demo, observe verified delivery + measured response, see a perturbation-driven
 * safe stop, and have the evidence verified locally — no backend. Also captures
 * the screenshots embedded in the project README.
 */

test("converging demo verifies locally", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".topbar .pill")).toHaveText("Not a medical device");

  await page.getByTestId("preset-relaxed").click();

  // Acceptance pass + all local checks green.
  await expect(page.getByText("ACCEPTANCE PASS")).toBeVisible();
  const checklist = page.getByTestId("checklist");
  await expect(checklist).toBeVisible();
  await expect(checklist.locator(".check-bad")).toHaveCount(0);

  await page.screenshot({ path: "screenshots/overview.png", fullPage: true });

  // Live session chart — move playback to a mid-session stimulating step.
  await page.getByTestId("nav-session").click();
  await expect(page.getByTestId("playback")).toBeVisible();
  const slider = page.locator('input[type=range]').first();
  await slider.fill("8");
  await page.screenshot({ path: "screenshots/session.png", fullPage: true });

  // Stimulus verifier shows verified receipts at that step.
  await page.getByTestId("nav-stimulus").click();
  await page.locator('input[type=range]').first().fill("8");
  await expect(page.getByTestId("receipt-table")).toBeVisible();
  await page.screenshot({ path: "screenshots/stimulus.png", fullPage: true });
});

test("perturbation triggers a visible fail-safe stop", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("preset-relaxed-safestop").click();

  await page.getByTestId("nav-safety").click();
  const breach = page.getByTestId("breach-box");
  await expect(breach).toBeVisible();
  await expect(breach.getByText("Fail-safe stop", { exact: false })).toBeVisible();
  await page.screenshot({ path: "screenshots/safety.png", fullPage: true });
});

test("audit trail chain verifies", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("preset-gamma").click();
  await page.getByTestId("nav-audit").click();
  await expect(page.getByText("CHAIN VALID")).toBeVisible();
  await expect(page.getByTestId("timeline")).toBeVisible();
  await page.screenshot({ path: "screenshots/audit.png", fullPage: true });
});
