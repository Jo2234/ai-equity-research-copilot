import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { buildDemoChatResponse, buildDemoMemo, demoCompanies, demoDocuments } from "./demoData";
import { App } from "./App";


function mockApi(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const url = String(input);
  const json = (payload: unknown) => Promise.resolve(new Response(JSON.stringify(payload), {
    status: 200, headers: { "Content-Type": "application/json" }
  }));
  if (url.endsWith("/companies")) return json(demoCompanies);
  if (url.includes("/companies/") && !init?.method) {
    const companyId = url.split("/").pop();
    return json({ documents: demoDocuments.filter((document) => document.company_id === companyId) });
  }
  const body = JSON.parse(String(init?.body ?? "{}"));
  if (url.endsWith("/research/chat")) return json(buildDemoChatResponse(body.question, ["cmp-nvda", "cmp-msft"]));
  if (url.endsWith("/research/memo")) return json(buildDemoMemo(demoCompanies.find((company) => company.id === body.company_id)!));
  if (url.endsWith("/research/compare")) return json({
    question: body.question,
    comparisons: demoCompanies.filter((company) => body.company_ids.includes(company.id)).map((company) => ({
      company: { ticker: company.ticker, name: company.name },
      summary: `${company.ticker} sourced summary`,
      key_points: [`${company.ticker} revenue driver [1].`],
      citations: buildDemoChatResponse(body.question, [company.id]).citations
    })),
    limitations: ["Mocked backend comparison contract."],
    usage: { model: "test", latency_ms: 1, input_tokens: 1, output_tokens: 1, estimated_cost_usd: 0 }
  });
  return Promise.resolve(new Response(JSON.stringify({ detail: "Mock route not configured" }), { status: 404 }));
}

beforeEach(() => { vi.stubGlobal("fetch", vi.fn(mockApi)); });
afterEach(() => { vi.unstubAllGlobals(); });

describe("AI Equity Research Copilot workstation", () => {
  it("renders companies and document readiness states", async () => {
    render(<App />);

    expect(await screen.findByText("NVIDIA Corporation")).toBeInTheDocument();
    expect(await screen.findByText("FY2025 Form 10-K")).toBeInTheDocument();
    expect((await screen.findAllByText("Ready")).length).toBeGreaterThan(0);
    expect(await screen.findByText("Processing")).toBeInTheDocument();
  });

  it("filters the local company list from the sidebar search", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("NVIDIA Corporation");
    await user.type(screen.getByLabelText("Search companies"), "msft");

    expect(screen.getByText("Microsoft Corporation")).toBeInTheDocument();
    expect(screen.queryByText("NVIDIA Corporation")).not.toBeInTheDocument();
  });

  it("validates required upload form fields", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("NVIDIA Corporation");
    await user.click(screen.getByRole("button", { name: /add to ingestion queue/i }));

    expect(screen.getByText("File, title, and document type are required.")).toBeInTheDocument();
  });

  it("submits a research question and opens citation evidence", async () => {
    const user = userEvent.setup();
    render(<App />);

    const question = await screen.findByLabelText("Research question");
    await user.clear(question);
    await user.type(question, "What drove data center growth?");
    await user.click(screen.getByRole("button", { name: /^ask$/i }));

    expect(await screen.findByText(/accelerated Data Center demand/i)).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: /FY2025 Form 10-K/i })[0]);

    await waitFor(() => {
      expect(screen.getAllByText(/Data Center revenue increased/i).length).toBeGreaterThan(0);
    });
  });

  it("shows failed document parse errors for the selected company", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("NVIDIA Corporation");
    await user.click(screen.getByRole("button", { name: /MSFT Microsoft Corporation/i }));

    expect(await screen.findByText("Cloud Segment Working Note")).toBeInTheDocument();
    expect(
      await screen.findByText("The uploaded text file appears to be empty. Upload a non-empty PDF or transcript.")
    ).toBeInTheDocument();
  });

  it("generates a memo and surfaces memo citations in the drawer", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("NVIDIA Corporation");
    await user.click(screen.getByRole("button", { name: /^memo$/i }));
    await user.click(screen.getByRole("button", { name: /generate memo/i }));

    expect(await screen.findByText(/NVDA Research Memo/i)).toBeInTheDocument();
    expect(screen.getByText("Risk factors")).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: /NVDA FY2025 Form 10-K/i })[0]);

    expect(await screen.findByText(/Data Center revenue increased primarily/i)).toBeInTheDocument();
  });

  it("runs a comparison for selected companies and renders sourced rows", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("NVIDIA Corporation");
    await user.click(screen.getByRole("button", { name: /^compare$/i }));
    await user.click(screen.getByRole("button", { name: /run comparison/i }));

    expect(await screen.findByRole("table", { name: /company comparison results/i })).toBeInTheDocument();
    expect(screen.getByText("NVDA revenue driver [1].")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "NVDA" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "MSFT" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "NVDA" }).length).toBeGreaterThan(0);
  });

  it("toggles retrieval debug and shows cited versus retrieved-only chunks after chat", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("NVIDIA Corporation");
    expect(screen.getByText("Run a chat request to inspect retrieved chunks.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /retrieval debug/i }));
    expect(screen.queryByText("Run a chat request to inspect retrieved chunks.")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /retrieval debug/i }));
    const question = screen.getByLabelText("Research question");
    await user.clear(question);
    await user.type(question, "Which chunks were retrieved?");
    await user.click(screen.getByRole("button", { name: /^ask$/i }));

    expect(await screen.findByText("3 chunks")).toBeInTheDocument();
    expect(screen.getByText(/retrieved only/i)).toBeInTheDocument();
  });
});


it("preserves a rejected upload for retry and shows the backend reason", async () => {
  const user = userEvent.setup();
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input).endsWith("/documents") && init?.method === "POST") {
      return Promise.resolve(new Response(JSON.stringify({ detail: "Upload exceeds the size limit" }), { status: 413 }));
    }
    return mockApi(input, init);
  }));
  render(<App />);
  await screen.findByText("NVIDIA Corporation");
  await user.type(screen.getByLabelText("Document title"), "Rejected report");
  await user.selectOptions(screen.getByLabelText("Document type"), "manual_note");
  const file = new File(["Contents"], "report.txt", { type: "text/plain" });
  await user.upload(screen.getByLabelText("Document file"), file);
  await user.click(screen.getByRole("button", { name: /add to ingestion queue/i }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Upload exceeds the size limit");
  expect(screen.getByLabelText("Document title")).toHaveValue("Rejected report");
  expect((screen.getByLabelText("Document file") as HTMLInputElement).files?.[0]).toBe(file);
  expect(screen.queryByText("Rejected report")).not.toBeInTheDocument();
});

it("shows a chat failure without substituting synthetic research", async () => {
  const user = userEvent.setup();
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input).endsWith("/research/chat")) {
      return Promise.resolve(new Response(JSON.stringify({ detail: "Research service unavailable" }), { status: 503 }));
    }
    return mockApi(input, init);
  }));
  render(<App />);
  await screen.findByText("NVIDIA Corporation");
  const question = screen.getByLabelText("Research question");
  await user.clear(question);
  await user.type(question, "How much cash does this company hold?");
  await user.click(screen.getByRole("button", { name: /^ask$/i }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Research service unavailable");
  expect(question).toHaveValue("How much cash does this company hold?");
  expect(screen.queryByText(/accelerated Data Center demand/i)).not.toBeInTheDocument();
  expect(screen.queryByText("Synthetic browser preview")).not.toBeInTheDocument();
});

it("does not populate a failed connection with demo companies", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Offline")));
  render(<App />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Workspace unavailable: Offline");
  expect(screen.queryByText("NVIDIA Corporation")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Retry connection" })).toBeInTheDocument();
});
