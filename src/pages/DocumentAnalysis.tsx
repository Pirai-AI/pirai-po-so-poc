import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Search, Loader2, ChevronRight, Eye, ArrowLeft, X, MessageCircle, Send, Bot, MinusCircle } from 'lucide-react';
import axios from 'axios';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

// Set pdf.js worker
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.js',
  import.meta.url,
).toString();

interface SearchResponse {
  query: string;
  results: any[];
  explanation: string;
  total_results: number;
}

interface InvoiceDetails {
  invoice_number: string;
  invoice_date: string;
  due_date: string;
  biller_company: string;
  biller_address: string;
  recipient_name: string;
  recipient_address: string;
  total_amount: number;
  currency: string;
  payment_terms: string;
  items: {
    item_name: string;
    quantity: number;
    unit_price: number;
    total_price: number;
    description: string;
  }[];
  tax_details: {
    tax_type: string;
    tax_rate: number;
    tax_amount: number;
    description: string;
  }[];
  subtotal: number;
  total_tax: number;
}

interface ChatMessage {
  type: 'user' | 'bot';
  content: string;
  results?: any[];
}

export default function DocumentAnalysis() {
  const { documentId } = useParams();
  const navigate = useNavigate();
  const [document, setDocument] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [invoiceDetails, setInvoiceDetails] = useState<InvoiceDetails | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isImage, setIsImage] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (documentId) {
      fetchDocument();
      fetchInvoiceDetails();
    }
  }, [documentId]);

  // Add scroll to bottom effect
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const fetchDocument = async () => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}${API_ENDPOINTS.GET_DOCUMENT_INFO(documentId!)}`
      );
      setDocument(response.data);
    } catch (err) {
      console.error('Error fetching document information:', err);
    }
  };

  const fetchInvoiceDetails = async () => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/invoice-details/${documentId}`
      );
      setInvoiceDetails(response.data);
    } catch (err) {
      console.error('Error fetching invoice details:', err);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    const query = searchQuery.trim();
    // Clear input immediately
    setSearchQuery('');
    
    // Add user message to chat
    setChatMessages(prev => [...prev, {
      type: 'user',
      content: query
    }]);
    
    setIsTyping(true);
    setError(null);
    
    try {
      const response = await axios.post(
        `${API_BASE_URL}${API_ENDPOINTS.SEARCH_GRAPH}`,
        {
          query: query,
          document_id: documentId
        }
      );
      
      // Add bot response to chat
      setChatMessages(prev => [...prev, {
        type: 'bot',
        content: response.data.explanation
      }]);
      
    } catch (err) {
      setError('Error searching the document. Please try again.');
    } finally {
      setIsTyping(false);
    }
  };

  const handlePreview = async () => {
    try {
      if (!document?.s3_key) return;
      
      // Check if file is an image
      const isImageFile = /\.(jpg|jpeg|png)$/i.test(document.s3_key);
      setIsImage(isImageFile);
      
      const response = await axios.get(
        `${API_BASE_URL}${API_ENDPOINTS.GET_DOCUMENT(document.s3_key)}`,
        { responseType: 'blob' }
      );
      
      // Clean up old URL if it exists
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl);
      }
      
      const fileUrl = URL.createObjectURL(response.data);
      setPdfUrl(fileUrl);
      setShowPreview(true);
      setPageNumber(1);
    } catch (err) {
      console.error('Error loading document preview:', err);
    }
  };

  // Add cleanup on component unmount
  useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl);
      }
    };
  }, [pdfUrl]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Title Bar */}
      <div className="bg-red-900 text-white">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={() => navigate('/')}
              className="flex items-center text-red-300 hover:text-red-200 transition-colors"
            >
              <ArrowLeft className="w-5 h-5 mr-2" />
              Back to Documents
            </button>
            <button
              onClick={handlePreview}
              className="flex items-center px-4 py-2 bg-red-600 rounded-lg hover:bg-red-700 transition-colors"
            >
              <Eye className="w-5 h-5 mr-2" />
              Preview Document
            </button>
          </div>
          <div className="mt-4">
            <h1 className="text-3xl font-['Poppins'] font-bold bg-gradient-to-r from-red-300 to-red-100 bg-clip-text text-transparent">
              Document Analysis
            </h1>
            <p className="text-red-200 text-sm mt-1">
              {document?.name || 'Loading document...'}
            </p>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {/* Invoice Details Section */}
        {invoiceDetails && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-6">Invoice Details</h2>
            
            {/* Basic Info Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div className="space-y-4">
                {(invoiceDetails.biller_company || invoiceDetails.biller_address) && (
                  <div className="bg-red-50 rounded-lg p-4">
                    <h3 className="text-red-800 font-medium mb-2">Biller Information</h3>
                    <div className="text-red-600">
                      {invoiceDetails.biller_company && (
                        <p className="font-medium">{invoiceDetails.biller_company}</p>
                      )}
                      {invoiceDetails.biller_address && (
                        <p className="whitespace-pre-line">{invoiceDetails.biller_address}</p>
                      )}
                    </div>
                  </div>
                )}
                {(invoiceDetails.recipient_name || invoiceDetails.recipient_address) && (
                  <div className="bg-red-50 rounded-lg p-4">
                    <h3 className="text-red-800 font-medium mb-2">Recipient Information</h3>
                    <div className="text-red-600">
                      {invoiceDetails.recipient_name && (
                        <p className="font-medium">{invoiceDetails.recipient_name}</p>
                      )}
                      {invoiceDetails.recipient_address && (
                        <p className="whitespace-pre-line">{invoiceDetails.recipient_address}</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
              <div className="bg-red-50 rounded-lg p-4">
                <h3 className="text-red-800 font-medium mb-4">Invoice Summary</h3>
                <div className="space-y-2 text-red-600">
                  {invoiceDetails.invoice_number && (
                    <div className="flex justify-between">
                      <span>Invoice Number:</span>
                      <span className="font-medium">{invoiceDetails.invoice_number}</span>
                    </div>
                  )}
                  {invoiceDetails.invoice_date && (
                    <div className="flex justify-between">
                      <span>Invoice Date:</span>
                      <span className="font-medium">
                        {new Date(invoiceDetails.invoice_date).toLocaleDateString('en-GB')}
                      </span>
                    </div>
                  )}
                  {invoiceDetails.due_date && (
                    <div className="flex justify-between">
                      <span>Due Date:</span>
                      <span className="font-medium">
                        {new Date(invoiceDetails.due_date).toLocaleDateString('en-GB')}
                      </span>
                    </div>
                  )}
                  {invoiceDetails.payment_terms && (
                    <div className="flex justify-between">
                      <span>Payment Terms:</span>
                      <span className="font-medium">{invoiceDetails.payment_terms}</span>
                    </div>
                  )}
                  {invoiceDetails.total_amount && (
                    <div className="flex justify-between text-lg font-medium mt-4 pt-4 border-t border-red-200">
                      <span>Total Amount:</span>
                      <span>{invoiceDetails.currency} {invoiceDetails.total_amount.toFixed(2)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Items Table */}
            {invoiceDetails.items && invoiceDetails.items.length > 0 && (
              <div className="mt-8">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Items</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-red-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-red-800 uppercase tracking-wider">Item</th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-red-800 uppercase tracking-wider">Quantity</th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-red-800 uppercase tracking-wider">Unit Price</th>
                        <th className="px-6 py-3 text-right text-xs font-medium text-red-800 uppercase tracking-wider">Total</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {invoiceDetails.items.map((item, index) => (
                        <tr key={index} className="hover:bg-red-50/30 transition-colors">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900 font-medium">{item.item_name}</div>
                            {item.description && (
                              <div className="text-xs text-gray-500">{item.description}</div>
                            )}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                            {item.quantity}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                            {invoiceDetails.currency} {item.unit_price.toFixed(2)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-gray-900">
                            {invoiceDetails.currency} {item.total_price.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                    {invoiceDetails.subtotal && (
                      <tfoot className="bg-red-50">
                        <tr>
                          <td colSpan={3} className="px-6 py-3 text-right text-sm font-medium text-red-800">Subtotal</td>
                          <td className="px-6 py-3 text-right text-sm font-medium text-red-800">
                            {invoiceDetails.currency} {invoiceDetails.subtotal.toFixed(2)}
                          </td>
                        </tr>
                      </tfoot>
                    )}
                  </table>
                </div>
              </div>
            )}

            {/* Tax Details */}
            {invoiceDetails.tax_details && invoiceDetails.tax_details.length > 0 && (
              <div className="mt-8">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Tax Details</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {invoiceDetails.tax_details.map((tax, index) => (
                    <div key={index} className="bg-red-50 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-red-800 font-medium">{tax.tax_type}</span>
                        {tax.tax_rate && <span className="text-red-800">{tax.tax_rate}%</span>}
                      </div>
                      {tax.tax_amount && (
                        <div className="flex justify-between items-center text-sm">
                          <span className="text-red-600">Amount</span>
                          <span className="text-red-800 font-medium">
                            {invoiceDetails.currency} {tax.tax_amount.toFixed(2)}
                          </span>
                        </div>
                      )}
                      {tax.description && (
                        <div className="mt-2 text-xs text-red-600">{tax.description}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Total Section */}
            {(invoiceDetails.subtotal || invoiceDetails.total_tax || invoiceDetails.total_amount) && (
              <div className="mt-8 border-t border-gray-200 pt-6">
                <div className="flex flex-col items-end space-y-2">
                  {invoiceDetails.subtotal && (
                    <div className="flex justify-between w-64">
                      <span className="text-gray-600">Subtotal</span>
                      <span className="text-gray-900 font-medium">
                        {invoiceDetails.currency} {invoiceDetails.subtotal.toFixed(2)}
                      </span>
                    </div>
                  )}
                  {invoiceDetails.total_tax && (
                    <div className="flex justify-between w-64">
                      <span className="text-gray-600">Total Tax</span>
                      <span className="text-gray-900 font-medium">
                        {invoiceDetails.currency} {invoiceDetails.total_tax.toFixed(2)}
                      </span>
                    </div>
                  )}
                  {invoiceDetails.total_amount && (
                    <div className="flex justify-between w-64 pt-2 border-t border-gray-200">
                      <span className="text-lg font-semibold text-gray-800">Total</span>
                      <span className="text-lg font-bold text-red-600">
                        {invoiceDetails.currency} {invoiceDetails.total_amount.toFixed(2)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Floating Chat Button */}
        <div className="fixed bottom-6 right-6 z-50">
          {!isChatOpen && (
            <button
              onClick={() => setIsChatOpen(true)}
              className="bg-red-600 hover:bg-red-700 text-white rounded-full p-4 shadow-lg transition-all duration-200 hover:scale-110"
            >
              <MessageCircle className="w-6 h-6" />
            </button>
          )}

          {/* Chat Interface */}
          {isChatOpen && (
            <div className="fixed bottom-6 right-6 w-96 h-[600px] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden border border-gray-200">
              {/* Chat Header */}
              <div className="bg-red-600 p-4 text-white flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Bot className="w-6 h-6" />
                  <div>
                    <h3 className="font-medium">Document Assistant</h3>
                    <p className="text-xs text-red-100">Ask me anything about this document</p>
                  </div>
                </div>
                <button
                  onClick={() => setIsChatOpen(false)}
                  className="text-red-100 hover:text-white transition-colors"
                >
                  <MinusCircle className="w-5 h-5" />
                </button>
              </div>

              {/* Chat Messages */}
              <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
                {chatMessages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl px-4 py-2 ${
                        message.type === 'user'
                          ? 'bg-red-600 text-white rounded-br-none'
                          : 'bg-gray-100 text-gray-800 rounded-bl-none'
                      }`}
                    >
                      <p className="text-sm">{message.content}</p>
                    </div>
                  </div>
                ))}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 rounded-2xl rounded-bl-none px-4 py-2">
                      <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <div className="p-4 border-t border-gray-200">
                <div className="relative">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Ask a question..."
                    className="w-full px-4 py-2 pr-12 border border-gray-200 rounded-full focus:ring-2 focus:ring-red-500 focus:border-transparent"
                    onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  />
                  <button
                    onClick={handleSearch}
                    disabled={isTyping || !searchQuery.trim()}
                    className={`absolute right-2 top-1/2 -translate-y-1/2 p-1.5 text-white rounded-full
                      ${isTyping || !searchQuery.trim()
                        ? 'bg-gray-300'
                        : 'bg-red-600 hover:bg-red-700'}`}
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
                {error && (
                  <p className="mt-2 text-xs text-red-500">{error}</p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* PDF Preview Modal */}
      {showPreview && pdfUrl && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 w-11/12 h-5/6 overflow-hidden flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-800">Document Preview</h2>
              <button
                onClick={() => setShowPreview(false)}
                className="p-2 text-gray-500 hover:text-gray-700 rounded-lg"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            <div className="flex-1 overflow-auto">
              {isImage ? (
                <img 
                  src={pdfUrl} 
                  alt="Document Preview" 
                  className="max-w-full h-auto mx-auto"
                />
              ) : (
                <Document
                  file={pdfUrl}
                  onLoadSuccess={({ numPages }) => setNumPages(numPages)}
                  className="max-w-full"
                >
                  <Page
                    pageNumber={pageNumber}
                    width={Math.min(window.innerWidth * 0.8, 1000)}
                    renderTextLayer={false}
                    renderAnnotationLayer={false}
                  />
                </Document>
              )}
            </div>
            {!isImage && numPages && numPages > 1 && (
              <div className="mt-4 text-center text-gray-600">
                Page {pageNumber} of {numPages}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
} 