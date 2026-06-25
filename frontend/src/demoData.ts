import type {
  ChatResponse,
  Company,
  CompareResponse,
  MemoResponse,
  ResearchDocument,
  RetrievalDebug,
  UploadPayload
} from "./types";

export const demoCompanies: Company[] = [
  {
    id: "cmp-nvda",
    ticker: "NVDA",
    name: "NVIDIA Corporation",
    exchange: "NASDAQ",
    sector: "Information Technology",
    industry: "Semiconductors"
  },
  {
    id: "cmp-msft",
    ticker: "MSFT",
    name: "Microsoft Corporation",
    exchange: "NASDAQ",
    sector: "Information Technology",
    industry: "Software"
  },
  {
    id: "cmp-amd",
    ticker: "AMD",
    name: "Advanced Micro Devices, Inc.",
    exchange: "NASDAQ",
    sector: "Information Technology",
    industry: "Semiconductors"
  }
];

export const demoDocuments: ResearchDocument[] = [
  {
    id: "doc-nvda-10k-2025",
    company_id: "cmp-nvda",
    title: "FY2025 Form 10-K",
    document_type: "10-k",
    filing_date: "2025-02-26",
    fiscal_year: 2025,
    status: "ready",
    chunk_count: 184
  },
  {
    id: "doc-nvda-q1-2026",
    company_id: "cmp-nvda",
    title: "Q1 FY2026 Earnings Transcript",
    document_type: "earnings_transcript",
    filing_date: "2025-05-28",
    fiscal_year: 2026,
    fiscal_quarter: 1,
    status: "processing",
    chunk_count: 48
  },
  {
    id: "doc-msft-10q-2026",
    company_id: "cmp-msft",
    title: "Q3 FY2026 Form 10-Q",
    document_type: "10-q",
    filing_date: "2026-04-25",
    fiscal_year: 2026,
    fiscal_quarter: 3,
    status: "ready",
    chunk_count: 121
  },
  {
    id: "doc-msft-note",
    company_id: "cmp-msft",
    title: "Cloud Segment Working Note",
    document_type: "manual_note",
    filing_date: "2026-05-06",
    fiscal_year: 2026,
    status: "failed",
    parse_error: "The uploaded text file appears to be empty. Upload a non-empty PDF or transcript."
  },
  {
    id: "doc-amd-presentation",
    company_id: "cmp-amd",
    title: "Investor Presentation",
    document_type: "investor_presentation",
    filing_date: "2026-03-12",
    fiscal_year: 2026,
    status: "uploaded",
    chunk_count: 0
  }
];

export const demoRetrieval = (query: string): RetrievalDebug => ({
  query,
  top_k: 8,
  threshold: 0.72,
  chunks: [
    {
      id: "chk-nvda-42",
      document_id: "doc-nvda-10k-2025",
      company_ticker: "NVDA",
      document_title: "FY2025 Form 10-K",
      section_title: "Management's Discussion and Analysis",
      page_start: 42,
      score: 0.89,
      excerpt:
        "Data Center revenue increased primarily from demand for the NVIDIA Hopper GPU computing platform used for training and inference of generative AI models.",
      cited: true
    },
    {
      id: "chk-nvda-risk-17",
      document_id: "doc-nvda-10k-2025",
      company_ticker: "NVDA",
      document_title: "FY2025 Form 10-K",
      section_title: "Risk Factors",
      page_start: 17,
      score: 0.81,
      excerpt:
        "Demand concentration, supply constraints, export controls, and rapid technology transitions may affect revenue timing and gross margin.",
      cited: true
    },
    {
      id: "chk-msft-cloud-9",
      document_id: "doc-msft-10q-2026",
      company_ticker: "MSFT",
      document_title: "Q3 FY2026 Form 10-Q",
      section_title: "Segment Results",
      page_start: 31,
      score: 0.76,
      excerpt:
        "Intelligent Cloud revenue increased due to growth in Azure and other cloud services, partially offset by higher infrastructure costs.",
      cited: false
    }
  ]
});

export const buildDemoChatResponse = (question: string): ChatResponse => {
  const retrieval = demoRetrieval(question);
  return {
    message_id: `demo-${Date.now()}`,
    answer:
      "The cited context points to accelerated Data Center demand as the main driver, especially GPU platforms used for AI training and inference. The answer should be treated as document-grounded but incomplete because the demo corpus does not include a full multi-quarter margin bridge.",
    key_points: [
      "Revenue growth is tied to Data Center demand for AI workloads.",
      "The cited risk language flags supply, export-control, and technology-transition constraints.",
      "A complete answer would require the latest 10-Q and earnings call transcript to reconcile margin and mix effects."
    ],
    confidence: "medium",
    limitations: ["Demo mode uses seeded excerpts until the backend returns live retrieval results."],
    citations: retrieval.chunks.filter((chunk) => chunk.cited).map((chunk) => ({
      label: `${chunk.company_ticker} ${chunk.document_title}, p. ${chunk.page_start}`,
      document_id: chunk.document_id,
      chunk_id: chunk.id,
      excerpt: chunk.excerpt,
      score: chunk.score,
      company_ticker: chunk.company_ticker,
      section_title: chunk.section_title,
      page_start: chunk.page_start,
      cited: chunk.cited
    })),
    usage: {
      model: "demo-grounded-response",
      latency_ms: 612,
      input_tokens: 1240,
      output_tokens: 318,
      estimated_cost_usd: 0
    },
    retrieval_debug: retrieval
  };
};

export const buildDemoMemo = (company: Company): MemoResponse => ({
  company: { ticker: company.ticker, name: company.name },
  business_summary:
    `${company.name} is shown in this demo workspace with document-grounded placeholders. Live mode should replace this section with retrieved filings and transcripts only.`,
  recent_performance: [
    "Revenue commentary should be tied to reported filings and management discussion.",
    "Margin analysis is pending the latest indexed quarter."
  ],
  growth_drivers: ["AI infrastructure demand", "Cloud platform adoption", "Product cycle execution"],
  margin_analysis: ["Mix shift and supply availability are the first items to verify with citations."],
  capital_allocation: ["No cited capital allocation conclusion is available in demo mode."],
  risk_factors: ["Supply constraints", "Export controls", "Demand concentration"],
  management_commentary: ["Upload or ingest the earnings transcript to ground management commentary."],
  bull_case: ["Sustained demand and operating leverage could support stronger fundamentals."],
  bear_case: ["Capacity constraints or weaker enterprise spending could pressure growth."],
  open_questions: [
    "What portion of growth is volume versus price/mix?",
    "How much incremental capex is required to support demand?"
  ],
  source_citations: buildDemoChatResponse("memo").citations,
  limitations: ["Generated from demo data because the backend was unavailable."]
});

export const buildDemoCompare = (companies: Company[], question: string): CompareResponse => ({
  question,
  summary:
    "Demo comparison separates company-specific claims and shows where citations would be attached. Connect the backend for full cross-company retrieval.",
  rows: [
    {
      dimension: "Growth driver",
      companies: Object.fromEntries(
        companies.map((company) => [
          company.ticker,
          company.ticker === "MSFT" ? "Cloud services and enterprise software demand." : "Accelerated compute and product-cycle demand."
        ])
      ),
      citations: buildDemoChatResponse(question).citations
    },
    {
      dimension: "Key risk",
      companies: Object.fromEntries(
        companies.map((company) => [
          company.ticker,
          company.ticker === "MSFT" ? "Infrastructure cost and platform competition." : "Supply, export-control, and demand concentration risk."
        ])
      ),
      citations: buildDemoChatResponse(question).citations.slice(0, 1)
    }
  ],
  usage: {
    model: "demo-compare-response",
    latency_ms: 744,
    input_tokens: 1512,
    output_tokens: 486,
    estimated_cost_usd: 0
  }
});

export const buildDemoUploadedDocument = (
  companyId: string,
  payload: UploadPayload
): ResearchDocument => ({
  id: `doc-upload-${Date.now()}`,
  company_id: companyId,
  title: payload.title,
  document_type: payload.document_type,
  filing_date: payload.filing_date,
  fiscal_year: payload.fiscal_year ? Number(payload.fiscal_year) : undefined,
  fiscal_quarter: payload.fiscal_quarter ? Number(payload.fiscal_quarter) : undefined,
  source_url: payload.source_url,
  status: "uploaded",
  chunk_count: 0
});
