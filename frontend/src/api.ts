import {
  buildDemoChatResponse, buildDemoCompare, buildDemoMemo,
  demoCompanies, demoDocuments
} from "./demoData";
import type {
  ChatResponse, Company, CompanyDiscoverResponse, CompanyLookupResult,
  CompareResponse, Citation, MemoResponse, ResearchDocument, UploadPayload
} from "./types";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "/api";
export const browserDemoMode = import.meta.env.VITE_BROWSER_DEMO === "true";

type QuestionParams = {
  companyIds: string[];
  question: string;
  documentTypes: string[];
  fiscalYears: number[];
  topK: number;
};

type BackendComparison = {
  question: string;
  comparisons: Array<{
    company: Pick<Company, "ticker" | "name">;
    summary: string;
    key_points: string[];
    citations: Citation[];
  }>;
  limitations?: string[];
  usage: CompareResponse["usage"];
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body: { detail?: unknown } = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = body.detail.map((item) => typeof item?.msg === "string" ? item.msg : "Invalid request").join("; ");
      }
    } catch {
      // Non-JSON error pages still retain their HTTP status.
    }
    throw new Error(`API ${response.status}: ${detail || "Request failed"}`);
  }
  return response.json() as Promise<T>;
}

const liveApi = {
  async fetchCompanies() {
    return { companies: await requestJson<Company[]>("/companies"), demo: false };
  },
  async searchCompanyUniverse(query: string) {
    const params = new URLSearchParams({ q: query, limit: "8" });
    const results = await requestJson<CompanyLookupResult[]>(`/companies/search?${params}`);
    return { results, demo: false };
  },
  async discoverCompany(query: string, formType = "10-k") {
    const discovery = await requestJson<CompanyDiscoverResponse>("/companies/discover", {
      method: "POST",
      body: JSON.stringify({ query, form_type: formType, build_corpus: true,
        annual_limit: 1, quarterly_limit: 4, current_report_limit: 6, proxy_limit: 1 })
    });
    return { discovery, demo: false };
  },
  async fetchDocuments(companyId: string) {
    const detail = await requestJson<{ documents: ResearchDocument[] }>(`/companies/${companyId}`);
    return { documents: detail.documents, demo: false };
  },
  async uploadDocument(companyId: string, payload: UploadPayload) {
    const formData = new FormData();
    formData.append("file", payload.file);
    formData.append("title", payload.title);
    formData.append("document_type", payload.document_type);
    if (payload.filing_date) formData.append("filing_date", payload.filing_date);
    if (payload.fiscal_year) formData.append("fiscal_year", payload.fiscal_year);
    if (payload.fiscal_quarter) formData.append("fiscal_quarter", payload.fiscal_quarter);
    if (payload.source_url) formData.append("source_url", payload.source_url);
    const document = await requestJson<ResearchDocument>(`/companies/${companyId}/documents`, {
      method: "POST", body: formData
    });
    return { document, demo: false };
  },
  async askResearchQuestion(params: QuestionParams) {
    const response = await requestJson<ChatResponse>("/research/chat", {
      method: "POST", body: JSON.stringify({ company_ids: params.companyIds,
        question: params.question, document_types: params.documentTypes,
        fiscal_years: params.fiscalYears, top_k: params.topK })
    });
    return { response: { ...response, citations: response.citations.map(normalizeCitation) }, demo: false };
  },
  async generateMemo(company: Company) {
    const memo = await requestJson<MemoResponse>("/research/memo", {
      method: "POST", body: JSON.stringify({ company_id: company.id })
    });
    return { memo: { ...memo, source_citations: memo.source_citations.map(normalizeCitation) }, demo: false };
  },
  async compareCompanies(companies: Company[], question: string) {
    const raw = await requestJson<BackendComparison>("/research/compare", {
      method: "POST", body: JSON.stringify({ company_ids: companies.map((company) => company.id),
        question, top_k_per_company: 5 })
    });
    return { comparison: normalizeComparison(raw, companies, question), demo: false };
  }
};

const demoApi: typeof liveApi = {
  async fetchCompanies() { return { companies: demoCompanies, demo: true }; },
  async searchCompanyUniverse(query) {
    const needle = query.trim().toLowerCase();
    const results = demoCompanies.filter((company) =>
      company.ticker.toLowerCase().includes(needle) || company.name.toLowerCase().includes(needle)
    ).map((company) => ({ ticker: company.ticker, name: company.name, source: "local",
      local_company_id: company.id, already_in_workspace: true }));
    return { results, demo: true };
  },
  async fetchDocuments(companyId) {
    return { documents: demoDocuments.filter((document) => document.company_id === companyId), demo: true };
  },
  async discoverCompany() { throw new Error("SEC imports require the live API; browser demo is read-only."); },
  async uploadDocument() { throw new Error("Uploads require the live API; browser demo is read-only."); },
  async askResearchQuestion(params) {
    return { response: buildDemoChatResponse(params.question, params.companyIds), demo: true };
  },
  async generateMemo(company) { return { memo: buildDemoMemo(company), demo: true }; },
  async compareCompanies(companies, question) {
    return { comparison: buildDemoCompare(companies, question), demo: true };
  }
};

// Mode is chosen once, before any requests. HTTP failure never changes adapters.
export const {
  fetchCompanies, searchCompanyUniverse, discoverCompany, fetchDocuments,
  uploadDocument, askResearchQuestion, generateMemo, compareCompanies
} = browserDemoMode ? demoApi : liveApi;

function normalizeCitation(citation: Citation): Citation {
  return {
    ...citation,
    company_ticker: citation.company_ticker ?? citation.label.split(" ")[0],
    title: citation.title ?? citation.label,
    page_start: citation.page_start ?? pageFromLabel(citation.label),
    cited: citation.cited ?? true
  };
}

function pageFromLabel(label: string): number | undefined {
  const match = label.match(/p\.\s*(\d+)/i);
  return match ? Number(match[1]) : undefined;
}

function normalizeComparison(
  raw: BackendComparison,
  companies: Company[],
  question: string
): CompareResponse {
  return {
    question: raw.question || question,
    summary: raw.limitations?.join(" ") || "Comparison is limited to ingested documents.",
    rows: raw.comparisons.map((item) => ({
      dimension: item.company.ticker,
      companies: Object.fromEntries(
        companies.map((company) => [
          company.ticker,
          company.ticker === item.company.ticker ? item.key_points.join(" ") || item.summary : ""
        ])
      ),
      citations: item.citations.map(normalizeCitation)
    })),
    usage: raw.usage
  };
}
