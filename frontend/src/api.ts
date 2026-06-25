import {
  buildDemoChatResponse,
  buildDemoCompare,
  buildDemoMemo,
  buildDemoUploadedDocument,
  demoCompanies,
  demoDocuments
} from "./demoData";
import type {
  ChatResponse,
  Company,
  CompanyDiscoverResponse,
  CompanyLookupResult,
  CompareResponse,
  Citation,
  MemoResponse,
  ResearchDocument,
  UploadPayload
} from "./types";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || "";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init
  });

  if (!response.ok) {
    throw new Error(`API ${response.status}: ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export async function fetchCompanies(): Promise<{ companies: Company[]; demo: boolean }> {
  try {
    const companies = await requestJson<Company[]>("/companies");
    return { companies, demo: false };
  } catch {
    return { companies: demoCompanies, demo: true };
  }
}

export async function searchCompanyUniverse(
  query: string
): Promise<{ results: CompanyLookupResult[]; demo: boolean }> {
  try {
    const params = new URLSearchParams({ q: query, limit: "8" });
    const results = await requestJson<CompanyLookupResult[]>(`/companies/search?${params.toString()}`);
    return { results, demo: false };
  } catch {
    const needle = query.trim().toLowerCase();
    return {
      results: demoCompanies
        .filter((company) => company.ticker.toLowerCase().includes(needle) || company.name.toLowerCase().includes(needle))
        .map((company) => ({
          ticker: company.ticker,
          name: company.name,
          source: "local",
          local_company_id: company.id,
          already_in_workspace: true
        })),
      demo: true
    };
  }
}

export async function discoverCompany(
  query: string,
  formType = "10-k"
): Promise<{ discovery: CompanyDiscoverResponse; demo: boolean }> {
  const discovery = await requestJson<CompanyDiscoverResponse>("/companies/discover", {
    method: "POST",
    body: JSON.stringify({ query, form_type: formType })
  });
  return { discovery, demo: false };
}

export async function fetchDocuments(companyId?: string): Promise<{ documents: ResearchDocument[]; demo: boolean }> {
  try {
    if (!companyId) {
      throw new Error("Company id is required until the backend exposes a document index endpoint.");
    }
    const detail = await requestJson<{ documents?: ResearchDocument[]; document_summary?: ResearchDocument[] }>(
      `/companies/${companyId}`
    );
    return { documents: detail.documents ?? detail.document_summary ?? [], demo: false };
  } catch {
    return {
      documents: companyId ? demoDocuments.filter((document) => document.company_id === companyId) : demoDocuments,
      demo: true
    };
  }
}

export async function uploadDocument(
  companyId: string,
  payload: UploadPayload
): Promise<{ document: ResearchDocument; demo: boolean }> {
  try {
    const formData = new FormData();
    formData.append("file", payload.file);
    formData.append("title", payload.title);
    formData.append("document_type", payload.document_type);
    if (payload.filing_date) formData.append("filing_date", payload.filing_date);
    if (payload.fiscal_year) formData.append("fiscal_year", payload.fiscal_year);
    if (payload.fiscal_quarter) formData.append("fiscal_quarter", payload.fiscal_quarter);
    if (payload.source_url) formData.append("source_url", payload.source_url);

    const document = await requestJson<ResearchDocument>(`/companies/${companyId}/documents`, {
      method: "POST",
      body: formData
    });
    return { document, demo: false };
  } catch {
    return { document: buildDemoUploadedDocument(companyId, payload), demo: true };
  }
}

export async function askResearchQuestion(params: {
  companyIds: string[];
  question: string;
  documentTypes: string[];
  fiscalYears: number[];
  topK: number;
}): Promise<{ response: ChatResponse; demo: boolean }> {
  try {
    const response = await requestJson<ChatResponse>("/research/chat", {
      method: "POST",
      body: JSON.stringify({
        company_ids: params.companyIds,
        question: params.question,
        document_types: params.documentTypes,
        fiscal_years: params.fiscalYears,
        top_k: params.topK
      })
    });
    const citations = response.citations.map(normalizeCitation);
    return {
      response: {
        ...response,
        citations,
        retrieval_debug:
          response.retrieval_debug ??
          buildRetrievalDebug(params.question, params.topK, citations)
      },
      demo: false
    };
  } catch {
    return { response: buildDemoChatResponse(params.question), demo: true };
  }
}

export async function generateMemo(company: Company): Promise<{ memo: MemoResponse; demo: boolean }> {
  try {
    const memo = await requestJson<MemoResponse>("/research/memo", {
      method: "POST",
      body: JSON.stringify({
        company_id: company.id,
        template: "standard_equity_research_memo"
      })
    });
    return {
      memo: {
        ...memo,
        source_citations: memo.source_citations.map(normalizeCitation)
      },
      demo: false
    };
  } catch {
    return { memo: buildDemoMemo(company), demo: true };
  }
}

export async function compareCompanies(
  companies: Company[],
  question: string
): Promise<{ comparison: CompareResponse; demo: boolean }> {
  try {
    const rawComparison = await requestJson<
      CompareResponse | {
        question: string;
        comparisons: Array<{
          company: Pick<Company, "ticker" | "name">;
          summary: string;
          key_points: string[];
          citations: Citation[];
        }>;
        limitations?: string[];
        usage: CompareResponse["usage"];
      }
    >("/research/compare", {
      method: "POST",
      body: JSON.stringify({
        company_ids: companies.map((company) => company.id),
        question,
        top_k_per_company: 5
      })
    });
    return { comparison: normalizeComparison(rawComparison, companies, question), demo: false };
  } catch {
    return { comparison: buildDemoCompare(companies, question), demo: true };
  }
}

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

function buildRetrievalDebug(query: string, topK: number, citations: Citation[]): ChatResponse["retrieval_debug"] {
  return {
    query,
    top_k: topK,
    threshold: 0,
    chunks: citations.map((citation) => ({
      id: citation.chunk_id,
      document_id: citation.document_id,
      company_ticker: citation.company_ticker ?? "Source",
      document_title: citation.title ?? citation.label,
      section_title: citation.section_title,
      page_start: citation.page_start,
      score: citation.score,
      excerpt: citation.excerpt,
      cited: true
    }))
  };
}

function normalizeComparison(
  raw: CompareResponse | {
    question: string;
    comparisons: Array<{
      company: Pick<Company, "ticker" | "name">;
      summary: string;
      key_points: string[];
      citations: Citation[];
    }>;
    limitations?: string[];
    usage: CompareResponse["usage"];
  },
  companies: Company[],
  question: string
): CompareResponse {
  if ("rows" in raw) {
    return {
      ...raw,
      rows: raw.rows.map((row) => ({
        ...row,
        citations: row.citations.map(normalizeCitation)
      }))
    };
  }

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
