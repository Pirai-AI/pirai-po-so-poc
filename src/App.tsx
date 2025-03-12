import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileText, Upload, Search, Database, AlertCircle, FileUp, Loader2, SearchIcon, ChevronRight, CheckCircle2 } from 'lucide-react';
import axios from 'axios';

interface SearchResponse {
  query: string;
  cypher_query: string;
  results: any[];
  total_results: number;
  explanation?: string;
  query_confidence?: number;
  token_counts?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [processComplete, setProcessComplete] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
      setError(null);
      setProcessComplete(false);
      setSearchResults(null);
    }
  });

  const handleFileUpload = async () => {
    if (!file) return;

    setProcessing(true);
    setError(null);
    setProcessComplete(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post('http://localhost:8000/process-document', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setProcessing(false);
      setProcessComplete(true);
      // Auto-focus the search input after processing
      const searchInput = document.getElementById('search-input');
      if (searchInput) searchInput.focus();
    } catch (err) {
      setError('Error processing document. Please try again.');
      setProcessing(false);
      setProcessComplete(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const response = await axios.post('http://localhost:8000/search-graph', {
        query: searchQuery
      });
      setSearchResults(response.data);
      setError(null);
    } catch (err) {
      setError('Error searching the database. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        <header className="text-center mb-12">
          <div className="inline-flex items-center justify-center space-x-3 mb-4">
            <FileText className="w-12 h-12 text-indigo-600" />
            <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-blue-600">
             Invoice Reader
            </h1>
          </div>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Intelligent document analysis powered by advanced AI
          </p>
        </header>

        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-gray-800 flex items-center">
              <FileUp className="w-8 h-8 mr-3 text-indigo-600" strokeWidth={1.5} />
              Document Upload
            </h2>
            {processComplete && (
              <div className="flex items-center text-emerald-600 bg-emerald-50 px-4 py-2 rounded-full animate-fadeIn">
                <CheckCircle2 className="w-5 h-5 mr-2" />
                <span className="font-medium">Processing Complete!</span>
              </div>
            )}
          </div>
          
          <div
            {...getRootProps()}
            className={`border-3 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200
              ${isDragActive 
                ? 'border-indigo-500 bg-indigo-50 scale-102' 
                : 'border-gray-300 hover:border-indigo-400 hover:bg-indigo-50/50'}`}
          >
            <input {...getInputProps()} />
            <Upload 
              className={`mx-auto h-16 w-16 mb-4 transition-colors duration-200
                ${isDragActive ? 'text-indigo-600' : 'text-gray-400'}`} 
              strokeWidth={1.5}
            />
            {file ? (
              <div className="space-y-2">
                <p className="text-lg font-medium text-indigo-600">{file.name}</p>
                <p className="text-sm text-gray-500">File selected and ready for processing</p>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-lg font-medium text-gray-700">
                  Drop your PDF invoice here
                </p>
                <p className="text-sm text-gray-500">
                  or click to browse from your computer
                </p>
              </div>
            )}
          </div>

          <button
            onClick={handleFileUpload}
            disabled={!file || processing}
            className={`mt-8 w-full py-4 px-6 rounded-xl font-medium text-white flex items-center justify-center space-x-3
              transform transition-all duration-200
              ${!file || processing
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-indigo-600 to-blue-600 hover:scale-102 hover:shadow-lg active:scale-98'}`}
          >
            {processing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Processing Document...</span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                <span>Process Document</span>
              </>
            )}
          </button>
        </div>

        {/* Search Section - Only visible after processing */}
        {processComplete && (
          <div className="bg-white rounded-2xl shadow-xl p-8 transform transition-all duration-500 animate-fadeIn">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
              <SearchIcon className="w-8 h-8 mr-3 text-indigo-600" strokeWidth={1.5} />
              Search Document
            </h2>

            <div className="mb-6">
              <div className="relative">
                <input
                  id="search-input"
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Ask anything about your document..."
                  className="w-full px-5 py-4 pr-36 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                />
                <button
                  onClick={handleSearch}
                  disabled={isSearching}
                  className={`absolute right-2 top-2 px-6 py-2 bg-gradient-to-r from-indigo-600 to-blue-600 
                    text-white rounded-lg flex items-center space-x-2 transform transition-all duration-200
                    hover:scale-102 hover:shadow-md active:scale-98 ${isSearching ? 'opacity-75' : ''}`}
                >
                  {isSearching ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Searching...</span>
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4" />
                      <span>Search</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {searchResults && (
              <div className="space-y-4 animate-fadeIn">
                <div className="p-6 bg-gradient-to-br from-gray-50 to-indigo-50/30 rounded-xl">
                  <h3 className="font-medium text-gray-800 mb-4 flex items-center">
                    <Database className="w-5 h-5 mr-2 text-indigo-600" />
                    Found {searchResults.total_results} results
                  </h3>
                  <div className="space-y-3">
                    {searchResults.results.map((result, index) => (
                      <div 
                        key={index} 
                        className="p-6 bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow duration-200"
                      >
                        {Object.entries(result).map(([key, value]) => (
                          <div key={key} className="mb-2">
                            <span className="font-medium text-gray-700">{key}: </span>
                            <span className="text-gray-600">{JSON.stringify(value)}</span>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>

                {searchResults.explanation && (
                  <div className="p-6 bg-blue-50 rounded-xl border border-blue-100">
                    <div className="flex items-start space-x-3">
                      <ChevronRight className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                      <div>
                        <strong className="text-blue-700 block mb-1">AI Explanation</strong>
                        <p className="text-blue-600">{searchResults.explanation}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="mt-8 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center text-red-700 animate-fadeIn">
            <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;