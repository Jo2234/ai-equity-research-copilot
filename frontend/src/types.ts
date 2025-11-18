export type DocumentStatus = "uploaded" | "processing" | "ready" | "failed";

export type WorkspaceMode = "chat" | "memo" | "compare";

export interface Company {
  id: string;
  ticker: string;
  name: string;
  exchange?: string;
  sector?: string;
  industry?: string;
}

export interface CompanyLookupResult {
  ticker: string;
  name: string;
  cik?: number;
  source: "local" | "sec" | string;
  local_company_id?: string;
  already_in_workspace: boolean;
}

export interface CompanyDiscoverResponse {
  company: Company & {
    document_count: number;
    ready_document_count: number;
    documents: ResearchDocument[];
  };
  imported_document: ResearchDocument;
  imported_documents: ResearchDocument[];
  source: string;
  cik: number;
  accession_number: string;
  accession_numbers: string[];
  reused_existing_count: number;
}

export interface ResearchDocument {
  id: string;
  company_id: string;
  title: string;
  document_type: string;
  filing_date?: string;
  fiscal_year?: number;
  fiscal_quarter?: number;
  status: DocumentStatus;
  chunk_count?: number;
  parse_error?: string;
  source_url?: string;
}

export interface Citation {
  label: string;
  document_id: string;
  chunk_id: string;
  excerpt: string;
  score: number;
  company_id?: string;
  company_ticker?: string;
  title?: string;
  section_title?: string;
  page_start?: number;
  cited?: boolean;
}

export interface Usage {
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  estimated_cost_usd: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  usage?: Usage;
  confidence?: "high" | "medium" | "low";
  limitations?: string[];
}

export interface ChatResponse {
  message_id: string;
  answer: string;
  key_points?: string[];
  confidence?: "high" | "medium" | "low";
  limitations?: string[];
  citations: Citation[];
  usage: Usage;
  retrieval_debug?: RetrievalDebug;
}

export interface MemoResponse {
  company: Pick<Company, "ticker" | "name">;
  business_summary: string;
  recent_performance: string[];
  growth_drivers: string[];
  margin_analysis: string[];
  capital_allocation: string[];
  risk_factors: string[];
  management_commentary: string[];
  bull_case: string[];
  bear_case: string[];
  open_questions: string[];
  source_citations: Citation[];
  limitations: string[];
}

export interface CompareRow {
  dimension: string;
  companies: Record<string, string>;
  citations: Citation[];
}

export interface CompareResponse {
  question: string;
  rows: CompareRow[];
  summary: string;
  usage: Usage;
}

export interface RetrievalChunk {
  id: string;
  document_id: string;
  company_ticker: string;
  document_title: string;
  section_title?: string;
  page_start?: number;
  score: number;
  excerpt: string;
  cited: boolean;
}

export interface RetrievalDebug {
  query: string;
  top_k: number;
  threshold: number;
  chunks: RetrievalChunk[];
}

export interface UploadPayload {
  file: File;
  title: string;
  document_type: string;
  filing_date?: string;
  fiscal_year?: string;
  fiscal_quarter?: string;
  source_url?: string;
}
