import { FormEvent, useEffect, useMemo, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  BookOpenText,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  Clock3,
  Code2,
  DatabaseZap,
  Download,
  FileCheck2,
  FileText,
  Filter,
  Gauge,
  GitCompareArrows,
  Layers3,
  Loader2,
  MessageSquareText,
  PanelRightOpen,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  TrendingUp,
  Upload,
  XCircle
} from "lucide-react";
import {
  askResearchQuestion,
  compareCompanies,
  discoverCompany,
  fetchCompanies,
  fetchDocuments,
  generateMemo,
  searchCompanyUniverse,
  uploadDocument
} from "./api";
import type {
  ChatMessage,
  Citation,
  Company,
  CompanyLookupResult,
  CompareResponse,
  MemoResponse,
  ResearchDocument,
  RetrievalDebug,
  UploadPayload,
  WorkspaceMode
} from "./types";

const documentTypeOptions = [
  "10-k",
  "10-q",
  "8-k",
  "earnings_transcript",
  "investor_presentation",
  "annual_report",
  "manual_note",
  "other"
];

const starterQuestion = "What drove recent revenue growth, and which risks should I verify before drafting a memo?";

const questionPresets = [
  "Bridge revenue growth to volume, price, mix, and segment commentary.",
  "List the most material risk-factor changes versus the prior filing.",
  "What evidence supports or weakens the margin expansion narrative?"
];

const memoTemplates = [
  "Standard equity memo",
  "Earnings reaction",
  "Risk-factor update",
  "Bull / bear frame"
];

const researchStages = [
  { label: "Corpus", detail: "documents indexed", icon: <DatabaseZap size={15} /> },
  { label: "Retrieve", detail: "top-k passages", icon: <Search size={15} /> },
  { label: "Synthesize", detail: "facts vs. interpretation", icon: <FileCheck2 size={15} /> },
  { label: "Audit", detail: "citations required", icon: <ShieldCheck size={15} /> }
];

const companySnapshots: Record<
  string,
  {
    metrics: Array<{ label: string; value: string; sublabel: string }>;
    focus: string[];
  }
> = {
  NVDA: {
    metrics: [
      { label: "Corpus", value: "184", sublabel: "ready chunks" },
      { label: "Coverage", value: "FY25", sublabel: "10-K indexed" },
      { label: "Watch", value: "AI demand", sublabel: "revenue driver" }
    ],
    focus: ["Data Center demand", "gross margin mix", "export-control risk"]
  },
  MSFT: {
    metrics: [
      { label: "Corpus", value: "121", sublabel: "ready chunks" },
      { label: "Coverage", value: "Q3 FY26", sublabel: "10-Q indexed" },
      { label: "Watch", value: "Azure", sublabel: "cloud growth" }
    ],
    focus: ["Azure growth", "infrastructure cost", "AI capex"]
  },
  AMD: {
    metrics: [
      { label: "Corpus", value: "0", sublabel: "chunks pending" },
      { label: "Coverage", value: "Deck", sublabel: "upload staged" },
      { label: "Watch", value: "MI300", sublabel: "accelerator cycle" }
    ],
    focus: ["accelerator demand", "competitive positioning", "client recovery"]
  }
};

export function App() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [selectedCompanyId, setSelectedCompanyId] = useState<string>("");
  const [documents, setDocuments] = useState<ResearchDocument[]>([]);
  const [mode, setMode] = useState<WorkspaceMode>("chat");
  const [apiMode, setApiMode] = useState<"live" | "demo">("live");
  const [question, setQuestion] = useState(starterQuestion);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Select a company, confirm documents are ready, then ask a grounded research question. Answers will show citations and retrieval metadata when available."
    }
  ]);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [retrievalDebug, setRetrievalDebug] = useState<RetrievalDebug | null>(null);
  const [debugVisible, setDebugVisible] = useState(true);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [isMemoLoading, setIsMemoLoading] = useState(false);
  const [memoTemplate, setMemoTemplate] = useState(memoTemplates[0]);
  const [memo, setMemo] = useState<MemoResponse | null>(null);
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [compareQuestion, setCompareQuestion] = useState(
    "Compare growth drivers, margin risk, and management commentary."
  );
  const [selectedCompareIds, setSelectedCompareIds] = useState<string[]>([]);
  const [isCompareLoading, setIsCompareLoading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [companyQuery, setCompanyQuery] = useState("");
  const [companyLookupResults, setCompanyLookupResults] = useState<CompanyLookupResult[]>([]);
  const [companySearchError, setCompanySearchError] = useState("");
  const [isCompanySearching, setIsCompanySearching] = useState(false);
  const [importingTicker, setImportingTicker] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    fetchCompanies().then(({ companies: nextCompanies, demo }) => {
      if (!mounted) return;
      setCompanies(nextCompanies);
      setSelectedCompanyId(nextCompanies[0]?.id ?? "");
      setSelectedCompareIds(nextCompanies.slice(0, 2).map((company) => company.id));
      setApiMode(demo ? "demo" : "live");
    });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedCompanyId) return;
    let mounted = true;
    fetchDocuments(selectedCompanyId).then(({ documents: nextDocuments, demo }) => {
      if (!mounted) return;
      setDocuments((current) => {
        const localPending = current.filter(
          (document) =>
            document.company_id === selectedCompanyId &&
            !nextDocuments.some((nextDocument) => nextDocument.id === document.id) &&
            (document.status === "uploaded" || document.status === "processing")
        );
        return [...localPending, ...nextDocuments];
      });
      if (demo) setApiMode("demo");
    });
    return () => {
      mounted = false;
    };
  }, [selectedCompanyId]);

  const selectedCompany = companies.find((company) => company.id === selectedCompanyId) ?? companies[0];
  const selectedCompareCompanies = companies.filter((company) => selectedCompareIds.includes(company.id));
  const visibleCompanies = companies.filter((company) => {
    const needle = companyQuery.trim().toLowerCase();
    if (!needle) return true;
    return company.ticker.toLowerCase().includes(needle) || company.name.toLowerCase().includes(needle);
  });
  const readyDocuments = documents.filter((document) => document.status === "ready");
  const processingDocuments = documents.filter((document) => document.status === "processing" || document.status === "uploaded");
  const failedDocuments = documents.filter((document) => document.status === "failed");
  const readinessPercent = documents.length ? Math.round((readyDocuments.length / documents.length) * 100) : 0;
  const activeCitations = useMemo(
    () => messages.flatMap((message) => message.citations ?? []),
    [messages]
  );
  const selectedDocumentTypes = [...new Set(documents.map((document) => document.document_type))];
  const selectedFiscalYears = [
    ...new Set(documents.map((document) => document.fiscal_year).filter((year): year is number => Boolean(year)))
  ];
  const selectedSnapshot = selectedCompany
    ? companySnapshots[selectedCompany.ticker] ?? {
        metrics: [
          { label: "Corpus", value: String(readyDocuments.reduce((sum, document) => sum + (document.chunk_count ?? 0), 0)), sublabel: "ready chunks" },
          { label: "Coverage", value: `${readyDocuments.length}/${documents.length}`, sublabel: "ready documents" },
          { label: "Watch", value: selectedCompany.sector ?? "N/A", sublabel: "sector context" }
        ],
        focus: ["document coverage", "margin drivers", "risk disclosures"]
      }
    : null;

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedCompany || !question.trim()) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question.trim()
    };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setIsChatLoading(true);

    const { response, demo } = await askResearchQuestion({
      companyIds: [selectedCompany.id],
      question: userMessage.content,
      documentTypes: selectedDocumentTypes,
      fiscalYears: selectedFiscalYears,
      topK: 8
    });

    const assistantMessage: ChatMessage = {
      id: response.message_id,
      role: "assistant",
      content: response.answer,
      citations: response.citations,
      usage: response.usage,
      confidence: response.confidence,
      limitations: response.limitations
    };

    setMessages((current) => [...current, assistantMessage]);
    setSelectedCitation(response.citations[0] ?? null);
    setRetrievalDebug(response.retrieval_debug ?? null);
    if (demo) setApiMode("demo");
    setIsChatLoading(false);
  }

  async function handleGenerateMemo() {
    if (!selectedCompany) return;
    setIsMemoLoading(true);
    const { memo: nextMemo, demo } = await generateMemo(selectedCompany);
    setMemo(nextMemo);
    setSelectedCitation(nextMemo.source_citations[0] ?? null);
    if (demo) setApiMode("demo");
    setIsMemoLoading(false);
  }

  function handleExportMemo() {
    if (!memo) return;
    const markdown = buildMemoMarkdown(memo, memoTemplate);
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${memo.company.ticker.toLowerCase()}-research-memo.md`;
    link.click();
    URL.revokeObjectURL(url);
  }

  async function handleCompare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (selectedCompareCompanies.length < 2 || !compareQuestion.trim()) return;
    setIsCompareLoading(true);
    const { comparison: nextComparison, demo } = await compareCompanies(selectedCompareCompanies, compareQuestion);
    setComparison(nextComparison);
    setSelectedCitation(nextComparison.rows[0]?.citations[0] ?? null);
    if (demo) setApiMode("demo");
    setIsCompareLoading(false);
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedCompany) return;

    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const file = form.get("file");
    const title = String(form.get("title") ?? "").trim();
    const documentType = String(form.get("document_type") ?? "");

    if (!(file instanceof File) || !file.name || !title || !documentType) {
      setUploadError("File, title, and document type are required.");
      return;
    }

    setUploadError("");
    setIsUploading(true);
    const payload: UploadPayload = {
      file,
      title,
      document_type: documentType,
      filing_date: String(form.get("filing_date") ?? ""),
      fiscal_year: String(form.get("fiscal_year") ?? ""),
      fiscal_quarter: String(form.get("fiscal_quarter") ?? ""),
      source_url: String(form.get("source_url") ?? "")
    };
    const { document, demo } = await uploadDocument(selectedCompany.id, payload);
    setDocuments((current) => [document, ...current]);
    if (demo) setApiMode("demo");
    setIsUploading(false);
    formElement.reset();
  }

  async function handleCompanySearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = companyQuery.trim();
    if (!query) return;
    setCompanySearchError("");
    setIsCompanySearching(true);
    const { results, demo } = await searchCompanyUniverse(query);
    setCompanyLookupResults(results);
    if (demo) setApiMode("demo");
    if (!results.length) setCompanySearchError(`No company matches found for "${query}".`);
    setIsCompanySearching(false);
  }

  async function handleImportCompany(result: CompanyLookupResult) {
    const query = result.ticker || result.name;
    if (result.local_company_id) {
      setSelectedCompanyId(result.local_company_id);
      return;
    }
    setCompanySearchError("");
    setImportingTicker(result.ticker);
    try {
      const { discovery } = await discoverCompany(query, "10-k");
      const importedCompany = discovery.company;
      setCompanies((current) => {
        const withoutExisting = current.filter((company) => company.id !== importedCompany.id);
        return [...withoutExisting, importedCompany].sort((a, b) => a.ticker.localeCompare(b.ticker));
      });
      setSelectedCompanyId(importedCompany.id);
      setSelectedCompareIds((current) => [...new Set([importedCompany.id, ...current])].slice(0, 4));
      setDocuments(discovery.company.documents ?? [discovery.imported_document]);
      setCompanyLookupResults((current) =>
        current.map((item) =>
          item.ticker === result.ticker
            ? { ...item, local_company_id: importedCompany.id, already_in_workspace: true }
            : item
        )
      );
      setApiMode("live");
    } catch (error) {
      setCompanySearchError(error instanceof Error ? error.message : "Unable to import company filing.");
    } finally {
      setImportingTicker(null);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Company and document navigation">
        <div className="brand-row">
          <div className="brand-mark">ER</div>
          <div>
            <h1>Equity Research Copilot</h1>
            <p>Grounded document workstation</p>
          </div>
        </div>

        <div className="sidebar-search">
          <Search size={16} aria-hidden="true" />
          <form onSubmit={handleCompanySearch}>
            <label htmlFor="company-search">Companies</label>
            <div className="company-search-row">
              <input
                id="company-search"
                aria-label="Search companies"
                onChange={(event) => setCompanyQuery(event.target.value)}
                placeholder="Search ticker or name..."
                value={companyQuery}
              />
              <button disabled={isCompanySearching || !companyQuery.trim()} type="submit">
                {isCompanySearching ? <Loader2 className="spin" size={14} /> : <Search size={14} />}
                SEC
              </button>
            </div>
          </form>
        </div>

        {companySearchError ? <p className="company-search-error">{companySearchError}</p> : null}
        {companyLookupResults.length ? (
          <div className="company-lookup-list" aria-label="Company search results">
            {companyLookupResults.map((result) => (
              <button
                className="company-lookup-row"
                key={`${result.source}-${result.ticker}-${result.cik ?? "local"}`}
                onClick={() => handleImportCompany(result)}
                type="button"
              >
                <span>
                  <strong>{result.ticker}</strong>
                  <small>{result.name}</small>
                </span>
                <em>{result.already_in_workspace ? "Open" : importingTicker === result.ticker ? "Importing" : "Import 10-K"}</em>
              </button>
            ))}
          </div>
        ) : null}

        <div className="company-list" aria-label="Company list">
          {visibleCompanies.map((company) => (
            <button
              className={`company-row ${company.id === selectedCompanyId ? "selected" : ""}`}
              key={company.id}
              onClick={() => setSelectedCompanyId(company.id)}
              type="button"
            >
              <span className="ticker">{company.ticker}</span>
              <span>
                <strong>{company.name}</strong>
                <small>{company.sector ?? "Unclassified"} · {company.exchange ?? "N/A"}</small>
              </span>
              <ChevronRight size={16} aria-hidden="true" />
            </button>
          ))}
          {!visibleCompanies.length ? <EmptyState text="No local workspace companies match this search. Use SEC search to import one." compact /> : null}
        </div>

        <section className="coverage-panel" aria-label="Research focus">
          <div className="section-title">
            <TrendingUp size={16} aria-hidden="true" />
            <h2>Research Focus</h2>
          </div>
          <div className="focus-list">
            {(selectedSnapshot?.focus ?? []).map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </section>

        <section className="upload-panel" aria-labelledby="upload-heading">
          <div className="section-title">
            <Upload size={16} aria-hidden="true" />
            <h2 id="upload-heading">Upload Document</h2>
          </div>
          <form onSubmit={handleUpload}>
            <input name="title" placeholder="Document title" aria-label="Document title" />
            <select name="document_type" defaultValue="" aria-label="Document type">
              <option value="" disabled>
                Document type
              </option>
              {documentTypeOptions.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
            <div className="form-grid">
              <input name="filing_date" type="date" aria-label="Filing date" />
              <input name="fiscal_year" placeholder="FY" inputMode="numeric" aria-label="Fiscal year" />
            </div>
            <input name="fiscal_quarter" placeholder="Quarter, e.g. 1" inputMode="numeric" aria-label="Fiscal quarter" />
            <input name="source_url" placeholder="Source URL" aria-label="Source URL" />
            <input name="file" type="file" accept=".pdf,.txt,.md" aria-label="Document file" />
            {uploadError ? <p className="form-error">{uploadError}</p> : null}
            <button className="primary-button" disabled={isUploading} type="submit">
              {isUploading ? <Loader2 className="spin" size={16} /> : <Plus size={16} />}
              Add to ingestion queue
            </button>
          </form>
        </section>

        <section className="document-panel" aria-labelledby="documents-heading">
          <div className="section-title">
            <FileText size={16} aria-hidden="true" />
            <h2 id="documents-heading">Documents</h2>
          </div>
          <div className="readiness-strip">
            <span>{readyDocuments.length} ready</span>
            <span>{documents.length} total</span>
          </div>
          <div className="document-health" aria-label="Document readiness summary">
            <MetricBadge label="Ready" value={readyDocuments.length} />
            <MetricBadge label="Queued" value={processingDocuments.length} />
            <MetricBadge label="Failed" value={failedDocuments.length} />
          </div>
          <div className="document-list">
            {documents.length ? (
              documents.map((document) => (
                <DocumentRow key={document.id} document={document} />
              ))
            ) : (
              <EmptyState text="Upload a filing, transcript, presentation, or note to create a searchable corpus." compact />
            )}
          </div>
        </section>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">Selected company</span>
            <h2>{selectedCompany ? `${selectedCompany.ticker} · ${selectedCompany.name}` : "Loading company universe"}</h2>
            <p>{selectedCompany ? `${selectedCompany.industry ?? selectedCompany.sector ?? "Coverage universe"} · ${readinessPercent}% corpus ready` : "No company selected"}</p>
          </div>
          <div className="topbar-controls">
            <StatusPill tone={apiMode === "live" ? "good" : "warn"}>
              <Gauge size={14} />
              {apiMode === "live" ? "API live" : "Demo fallback"}
            </StatusPill>
            <StatusPill tone={readyDocuments.length ? "good" : "warn"}>
              <CheckCircle2 size={14} />
              {readyDocuments.length ? "Ready corpus" : "No ready docs"}
            </StatusPill>
            <button className="ghost-button" onClick={() => setDebugVisible((visible) => !visible)} type="button">
              <Code2 size={16} />
              Retrieval debug
            </button>
          </div>
        </header>

        <section className="market-strip" aria-label="Company research snapshot">
          {(selectedSnapshot?.metrics ?? []).map((metric) => (
            <MetricCard key={metric.label} label={metric.label} value={metric.value} sublabel={metric.sublabel} />
          ))}
          <div className="risk-note">
            <ShieldCheck size={16} aria-hidden="true" />
            <span>No investment advice. Factual claims require retrieved evidence.</span>
          </div>
        </section>

        <section className="workflow-strip" aria-label="Research workflow">
          {researchStages.map((stage, index) => (
            <div className="workflow-step" key={stage.label}>
              <span>{stage.icon}</span>
              <div>
                <strong>{index + 1}. {stage.label}</strong>
                <small>{stage.detail}</small>
              </div>
            </div>
          ))}
        </section>

        <div className="filter-bar" aria-label="Active filters">
          <Filter size={16} aria-hidden="true" />
          <FilterChip label="Types" value={selectedDocumentTypes.join(", ") || "none"} />
          <FilterChip label="Fiscal years" value={selectedFiscalYears.join(", ") || "all"} />
          <FilterChip label="Top K" value="8" />
          <FilterChip label="Citation policy" value="Required for factual claims" />
        </div>

        <nav className="mode-tabs" aria-label="Workspace mode">
          <ModeButton active={mode === "chat"} icon={<MessageSquareText size={16} />} onClick={() => setMode("chat")}>
            Chat
          </ModeButton>
          <ModeButton active={mode === "memo"} icon={<ClipboardList size={16} />} onClick={() => setMode("memo")}>
            Memo
          </ModeButton>
          <ModeButton active={mode === "compare"} icon={<Layers3 size={16} />} onClick={() => setMode("compare")}>
            Compare
          </ModeButton>
        </nav>

        {mode === "chat" ? (
          <section className="work-panel" aria-label="Research chat">
            <div className="panel-heading">
              <div>
                <h3>Grounded Q&A</h3>
                <p>{readyDocuments.length} ready documents · {activeCitations.length} active citations · top 8 retrieval</p>
              </div>
              <StatusPill tone={readyDocuments.length ? "good" : "warn"}>
                <BookOpenText size={14} />
                {readyDocuments.length ? "Evidence available" : "Evidence pending"}
              </StatusPill>
            </div>
            <div className="chat-log">
              {messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  <div className="message-header">
                    <strong>{message.role === "user" ? "Analyst" : "Copilot"}</strong>
                    {message.confidence ? <span className="confidence">{message.confidence} confidence</span> : null}
                  </div>
                  <p>{message.content}</p>
                  {message.citations?.length ? (
                    <div className="citation-buttons">
                      {message.citations.map((citation) => (
                        <button key={citation.chunk_id} onClick={() => setSelectedCitation(citation)} type="button">
                          <BookOpenText size={14} />
                          {citation.label}
                        </button>
                      ))}
                    </div>
                  ) : null}
                  {message.usage ? (
                    <div className="usage-row">
                      <span>{message.usage.model}</span>
                      <span>{message.usage.latency_ms} ms</span>
                      <span>${message.usage.estimated_cost_usd.toFixed(4)}</span>
                    </div>
                  ) : null}
                  {message.limitations?.length ? (
                    <ul className="limitations">
                      {message.limitations.map((limitation) => (
                        <li key={limitation}>{limitation}</li>
                      ))}
                    </ul>
                  ) : null}
                </article>
              ))}
              {isChatLoading ? (
                <article className="message assistant loading">
                  <Loader2 className="spin" size={16} />
                  Retrieving passages and drafting a cited answer
                </article>
              ) : null}
            </div>
            <form className="chat-input" onSubmit={handleAsk}>
              <div className="prompt-strip" aria-label="Question presets">
                {questionPresets.map((preset) => (
                  <button key={preset} onClick={() => setQuestion(preset)} type="button">
                    {preset}
                  </button>
                ))}
              </div>
              <textarea
                aria-label="Research question"
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask a cited research question..."
                value={question}
              />
              <button className="primary-button" disabled={!question.trim() || isChatLoading} type="submit">
                <ArrowUpRight size={16} />
                Ask
              </button>
            </form>
          </section>
        ) : null}

        {mode === "memo" ? (
          <section className="work-panel memo-panel" aria-label="Research memo generator">
            <div className="panel-heading">
              <div>
                <h3>Standard Research Memo</h3>
                <p>{memoTemplate} · editable analyst-style sections with source citations.</p>
              </div>
              <div className="panel-actions">
                <select
                  aria-label="Memo template"
                  onChange={(event) => setMemoTemplate(event.target.value)}
                  value={memoTemplate}
                >
                  {memoTemplates.map((template) => (
                    <option key={template} value={template}>
                      {template}
                    </option>
                  ))}
                </select>
                {memo ? (
                  <button className="ghost-button" onClick={handleExportMemo} type="button">
                    <Download size={16} />
                    Export
                  </button>
                ) : null}
                <button className="primary-button" disabled={isMemoLoading || !selectedCompany} onClick={handleGenerateMemo} type="button">
                  {isMemoLoading ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
                  Generate memo
                </button>
              </div>
            </div>
            {memo ? <MemoView memo={memo} onCitation={setSelectedCitation} /> : <EmptyState text="Generate a memo after documents are indexed and ready." />}
          </section>
        ) : null}

        {mode === "compare" ? (
          <section className="work-panel compare-panel" aria-label="Company comparison">
            <div className="panel-heading">
              <div>
                <h3>Company Comparison</h3>
                <p>{selectedCompareCompanies.length} selected companies · claims separated by ticker.</p>
              </div>
              <StatusPill tone={selectedCompareCompanies.length >= 2 ? "good" : "warn"}>
                <GitCompareArrows size={14} />
                {selectedCompareCompanies.length >= 2 ? "Ready to compare" : "Select two or more"}
              </StatusPill>
            </div>
            <form className="compare-form" onSubmit={handleCompare}>
              <div className="compare-company-grid">
                {companies.map((company) => (
                  <label className="checkbox-row" key={company.id}>
                    <input
                      checked={selectedCompareIds.includes(company.id)}
                      onChange={(event) => {
                        setSelectedCompareIds((current) =>
                          event.target.checked
                            ? [...current, company.id]
                            : current.filter((companyId) => companyId !== company.id)
                        );
                      }}
                      type="checkbox"
                    />
                    {company.ticker}
                  </label>
                ))}
              </div>
              <textarea
                aria-label="Comparison question"
                onChange={(event) => setCompareQuestion(event.target.value)}
                value={compareQuestion}
              />
              <button
                className="primary-button"
                disabled={selectedCompareCompanies.length < 2 || !compareQuestion.trim() || isCompareLoading}
                type="submit"
              >
                {isCompareLoading ? <Loader2 className="spin" size={16} /> : <Layers3 size={16} />}
                Run comparison
              </button>
            </form>
            {comparison ? (
              <ComparisonView comparison={comparison} tickers={selectedCompareCompanies.map((company) => company.ticker)} onCitation={setSelectedCitation} />
            ) : (
              <EmptyState text="Select at least two companies and run a sourced comparison." />
            )}
          </section>
        ) : null}

        {debugVisible ? <RetrievalDebugPanel debug={retrievalDebug} /> : null}
      </main>

      <aside className="citation-drawer" aria-label="Citation drawer">
        <div className="drawer-heading">
          <div>
            <span className="eyebrow">Source audit</span>
            <h2>Citations</h2>
          </div>
          <PanelRightOpen size={18} aria-hidden="true" />
        </div>
        <div className="audit-summary">
          <MetricBadge label="Claims" value={activeCitations.length} />
          <MetricBadge label="Ready docs" value={readyDocuments.length} />
          <MetricBadge label="Score" value={selectedCitation ? `${Math.round(selectedCitation.score * 100)}%` : "N/A"} />
        </div>
        {selectedCitation ? (
          <CitationDetail citation={selectedCitation} />
        ) : activeCitations.length ? (
          <CitationDetail citation={activeCitations[0]} />
        ) : (
          <EmptyState text="Click a citation to inspect the source excerpt, score, page, and section." />
        )}
        {activeCitations.length ? (
          <div className="drawer-list">
            {activeCitations.map((citation) => (
              <button key={citation.chunk_id} onClick={() => setSelectedCitation(citation)} type="button">
                <span>{citation.label}</span>
                <small>{Math.round(citation.score * 100)}% relevance</small>
              </button>
            ))}
          </div>
        ) : null}
      </aside>
    </div>
  );
}

function DocumentRow({ document }: { document: ResearchDocument }) {
  const statusMap = {
    ready: { icon: <CheckCircle2 size={15} />, label: "Ready", className: "ready" },
    processing: { icon: <Loader2 className="spin" size={15} />, label: "Processing", className: "processing" },
    uploaded: { icon: <Clock3 size={15} />, label: "Uploaded", className: "uploaded" },
    failed: { icon: <XCircle size={15} />, label: "Failed", className: "failed" }
  } as const;
  const status = statusMap[document.status];

  return (
    <article className="document-row">
      <div>
        <strong>{document.title}</strong>
        <small>
          {document.document_type} · {document.fiscal_year ?? "FY n/a"}
          {document.fiscal_quarter ? ` Q${document.fiscal_quarter}` : ""}
          {document.filing_date ? ` · filed ${document.filing_date}` : ""}
        </small>
      </div>
      <span className={`doc-status ${status.className}`}>
        {status.icon}
        {status.label}
      </span>
      {document.parse_error ? (
        <p className="parse-error">
          <AlertTriangle size={14} />
          {document.parse_error}
        </p>
      ) : (
        <p className="chunk-count">{document.chunk_count ?? 0} chunks indexed</p>
      )}
      {document.source_url ? (
        <a className="source-link" href={document.source_url} rel="noreferrer" target="_blank">
          Source
          <ArrowUpRight size={12} />
        </a>
      ) : null}
    </article>
  );
}

function StatusPill({ children, tone }: { children: ReactNode; tone: "good" | "warn" }) {
  return <span className={`status-pill ${tone}`}>{children}</span>;
}

function MetricBadge({ label, value }: { label: string; value: string | number }) {
  return (
    <span className="metric-badge">
      <strong>{value}</strong>
      {label}
    </span>
  );
}

function MetricCard({ label, sublabel, value }: { label: string; sublabel: string; value: string }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{sublabel}</small>
    </article>
  );
}

function FilterChip({ label, value }: { label: string; value: string }) {
  return (
    <span className="filter-chip">
      <strong>{label}</strong>
      {value}
    </span>
  );
}

function ModeButton({
  active,
  children,
  icon,
  onClick
}: {
  active: boolean;
  children: ReactNode;
  icon: ReactNode;
  onClick: () => void;
}) {
  return (
    <button className={active ? "active" : ""} onClick={onClick} type="button">
      {icon}
      {children}
    </button>
  );
}

function CitationDetail({ citation }: { citation: Citation }) {
  return (
    <article className="citation-detail">
      <h3>{citation.label}</h3>
      <dl>
        <div>
          <dt>Score</dt>
          <dd>{Math.round(citation.score * 100)}%</dd>
        </div>
        <div>
          <dt>Section</dt>
          <dd>{citation.section_title ?? "Not provided"}</dd>
        </div>
        <div>
          <dt>Page</dt>
          <dd>{citation.page_start ?? "N/A"}</dd>
        </div>
        <div>
          <dt>Chunk</dt>
          <dd>{citation.chunk_id}</dd>
        </div>
        <div>
          <dt>Company</dt>
          <dd>{citation.company_ticker ?? "N/A"}</dd>
        </div>
        <div>
          <dt>Document</dt>
          <dd>{citation.document_id}</dd>
        </div>
      </dl>
      <blockquote>{citation.excerpt}</blockquote>
    </article>
  );
}

function MemoView({ memo, onCitation }: { memo: MemoResponse; onCitation: (citation: Citation) => void }) {
  const sections = [
    ["Recent performance", memo.recent_performance],
    ["Growth drivers", memo.growth_drivers],
    ["Margin analysis", memo.margin_analysis],
    ["Capital allocation", memo.capital_allocation],
    ["Risk factors", memo.risk_factors],
    ["Management commentary", memo.management_commentary],
    ["Bull case", memo.bull_case],
    ["Bear case", memo.bear_case],
    ["Open questions", memo.open_questions],
    ["Limitations", memo.limitations]
  ] as const;

  return (
    <article className="memo-output">
      <h3>
        {memo.company.ticker} Research Memo <span>{memo.company.name}</span>
      </h3>
      <p>{memo.business_summary}</p>
      <div className="memo-grid">
        {sections.map(([title, points]) => (
          <section key={title}>
            <h4>{title}</h4>
            <ul>
              {points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          </section>
        ))}
      </div>
      <div className="citation-buttons">
        {memo.source_citations.map((citation) => (
          <button key={citation.chunk_id} onClick={() => onCitation(citation)} type="button">
            <BookOpenText size={14} />
            {citation.label}
          </button>
        ))}
      </div>
    </article>
  );
}

function ComparisonView({
  comparison,
  onCitation,
  tickers
}: {
  comparison: CompareResponse;
  onCitation: (citation: Citation) => void;
  tickers: string[];
}) {
  const gridStyle: CSSProperties = {
    gridTemplateColumns: `130px repeat(${Math.max(tickers.length, 1)}, minmax(140px, 1fr)) 120px`
  };

  return (
    <article className="comparison-output">
      <p>{comparison.summary}</p>
      <div className="comparison-table" role="table" aria-label="Company comparison results">
        <div className="comparison-header" role="row" style={gridStyle}>
          <strong role="columnheader">Dimension</strong>
          {tickers.map((ticker) => (
            <strong key={ticker} role="columnheader">
              {ticker}
            </strong>
          ))}
          <strong role="columnheader">Sources</strong>
        </div>
        {comparison.rows.map((row) => (
          <div className="comparison-row" key={row.dimension} role="row" style={gridStyle}>
            <strong role="cell">{row.dimension}</strong>
            {tickers.map((ticker) => (
              <span key={ticker} role="cell">
                {row.companies[ticker] ?? "No cited context"}
              </span>
            ))}
            <span className="source-cell" role="cell">
              {row.citations.map((citation) => (
                <button key={citation.chunk_id} onClick={() => onCitation(citation)} type="button">
                  {citation.company_ticker ?? "Source"}
                </button>
              ))}
            </span>
          </div>
        ))}
      </div>
    </article>
  );
}

function RetrievalDebugPanel({ debug }: { debug: RetrievalDebug | null }) {
  return (
    <section className="debug-panel" aria-label="Retrieval debug">
      <div className="panel-heading compact">
        <div>
          <h3>Retrieval Debug</h3>
          <p>{debug ? `Query: ${debug.query}` : "Run a chat request to inspect retrieved chunks."}</p>
        </div>
        {debug ? <span>{debug.chunks.length} chunks</span> : null}
      </div>
      {debug ? (
        <div className="debug-list">
          {debug.chunks.map((chunk) => (
            <article key={chunk.id} className={chunk.cited ? "cited" : ""}>
              <div>
                <strong>{chunk.company_ticker} · {chunk.document_title}</strong>
                <span>{Math.round(chunk.score * 100)}% · {chunk.cited ? "cited" : "retrieved only"}</span>
              </div>
              <p>{chunk.excerpt}</p>
              <small>{chunk.section_title ?? "No section"} · p. {chunk.page_start ?? "N/A"}</small>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function EmptyState({ compact = false, text }: { compact?: boolean; text: string }) {
  return <div className={`empty-state ${compact ? "compact" : ""}`}>{text}</div>;
}

function buildMemoMarkdown(memo: MemoResponse, template: string) {
  const sections: Array<[string, string[]]> = [
    ["Recent performance", memo.recent_performance],
    ["Growth drivers", memo.growth_drivers],
    ["Margin analysis", memo.margin_analysis],
    ["Capital allocation", memo.capital_allocation],
    ["Risk factors", memo.risk_factors],
    ["Management commentary", memo.management_commentary],
    ["Bull case", memo.bull_case],
    ["Bear case", memo.bear_case],
    ["Open questions", memo.open_questions],
    ["Limitations", memo.limitations]
  ];

  const body = sections
    .map(([title, points]) => [`## ${title}`, ...points.map((point) => `- ${point}`)].join("\n"))
    .join("\n\n");

  const citations = memo.source_citations
    .map((citation) => `- ${citation.label}: ${citation.excerpt}`)
    .join("\n");

  return `# ${memo.company.ticker} Research Memo\n\nTemplate: ${template}\nCompany: ${memo.company.name}\n\n${memo.business_summary}\n\n${body}\n\n## Source citations\n${citations || "- No citations returned."}\n`;
}
