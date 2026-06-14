import { sha256Hex } from "../verifier/verify";
import type { DeviceEvent } from "./link";

/**
 * A live, tamper-evident hash chain over real-mode device events — the same
 * construction as the offline evidence bundle, computed in the browser as
 * events arrive. Every connect / validate / stimulate / estop is recorded so a
 * real session leaves replayable, verifiable evidence (ADR-0014 §7.3, §11).
 */

const GENESIS = "0".repeat(64);

export interface DeviceAuditRecord {
  event: DeviceEvent;
  payloadSha256: string;
  prevHash: string;
  hash: string;
}

export function canonicalDevicePayload(e: DeviceEvent): string {
  return `${e.t}|${e.kind}|${e.detail}`;
}

export class DeviceAuditChain {
  private records: DeviceAuditRecord[] = [];

  get all(): readonly DeviceAuditRecord[] {
    return this.records;
  }

  get head(): string {
    return this.records.length ? this.records[this.records.length - 1].hash : GENESIS;
  }

  append(event: DeviceEvent): DeviceAuditRecord {
    const prevHash = this.head;
    const payloadSha256 = sha256Hex(canonicalDevicePayload(event));
    const hash = sha256Hex(prevHash + payloadSha256);
    const record = { event, payloadSha256, prevHash, hash };
    this.records.push(record);
    return record;
  }

  verify(): boolean {
    let prev = GENESIS;
    for (const r of this.records) {
      if (r.prevHash !== prev) return false;
      if (r.payloadSha256 !== sha256Hex(canonicalDevicePayload(r.event))) return false;
      if (r.hash !== sha256Hex(prev + r.payloadSha256)) return false;
      prev = r.hash;
    }
    return true;
  }
}
