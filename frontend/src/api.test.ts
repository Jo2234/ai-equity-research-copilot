import { afterEach, beforeEach, expect, it, vi } from "vitest";

beforeEach(() => { vi.resetModules(); vi.stubEnv("VITE_BROWSER_DEMO", "false"); });
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

it("propagates rejected real requests and keeps backend validation details", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: [{ msg: "Fiscal quarter must be at most 4" }] }), { status: 422 })));
  const api = await import("./api");
  await expect(api.uploadDocument("real-company", {
    file: new File(["Report"], "report.txt"), title: "Report", document_type: "manual_note", fiscal_quarter: "5"
  })).rejects.toThrow("API 422: Fiscal quarter must be at most 4");
});

it("does not fabricate retrieval audit metadata for a live response", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ answer: "Answer", citations: [], message_id: "real", usage: {} }))));
  const api = await import("./api");
  const result = await api.askResearchQuestion({ companyIds: ["real-company"], question: "Question", documentTypes: [], fiscalYears: [], topK: 8 });
  expect(result.demo).toBe(false);
  expect(result.response.retrieval_debug).toBeUndefined();
});

it("requires explicit browser demo mode and never sends requests in that mode", async () => {
  vi.stubEnv("VITE_BROWSER_DEMO", "true");
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  const api = await import("./api");
  const { companies, demo } = await api.fetchCompanies();
  expect(demo).toBe(true);
  expect(companies.length).toBeGreaterThan(0);
  const answer = await api.askResearchQuestion({ companyIds: ["cmp-msft"], question: "Cash?", documentTypes: [], fiscalYears: [], topK: 8 });
  expect(answer.response.citations).toEqual([]);
  expect(answer.response.answer).not.toContain("Data Center demand");
  await expect(api.uploadDocument("cmp-msft", { file: new File(["Report"], "report.txt"), title: "Report", document_type: "manual_note" })).rejects.toThrow("browser demo is read-only");
  expect(fetch).not.toHaveBeenCalled();
});
