import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { App } from "./App";

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
    expect(screen.getByText("Growth driver")).toBeInTheDocument();
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
