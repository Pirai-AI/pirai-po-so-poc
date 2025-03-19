export const API_BASE_URL = 'http://localhost:8000';

export const API_ENDPOINTS = {
  // Document Management
  DOCUMENTS: '/documents',
  PROCESS_DOCUMENT: '/process-document',
  DELETE_DOCUMENT: (documentId: string) => `/document/${documentId}`,
  GET_DOCUMENT: (s3Key: string) => `/document/${encodeURIComponent(s3Key)}`,
  GET_DOCUMENT_INFO: (documentId: string) => `/document-info/${documentId}`,
  
  // Search
  SEARCH_GRAPH: '/search-graph',
  GET_INVOICE_DETAILS: (documentId: string) => `/invoice-details/${documentId}`,
};

export const apiClient = {
  baseURL: API_BASE_URL,
  endpoints: API_ENDPOINTS,
}; 