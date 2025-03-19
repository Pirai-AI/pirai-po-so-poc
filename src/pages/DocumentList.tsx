import { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileText, Upload, AlertCircle, FileUp, Loader2, CheckCircle2, Trash2, Search } from 'lucide-react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

interface Document {
  id: string;
  name: string;
  original_name: string;
  s3_key: string;
}

export default function DocumentList() {
  const [file, setFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [processComplete, setProcessComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const navigate = useNavigate();

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png']
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
      setError(null);
      setProcessComplete(false);
    }
  });

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}${API_ENDPOINTS.DOCUMENTS}`);
      setDocuments(response.data);
    } catch (err) {
      setError('Error fetching documents');
    }
  };

  const handleFileUpload = async () => {
    if (!file) return;

    setProcessing(true);
    setError(null);
    setProcessComplete(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post(
        `${API_BASE_URL}${API_ENDPOINTS.PROCESS_DOCUMENT}`, 
        formData, 
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );
      setProcessing(false);
      setProcessComplete(true);
      await fetchDocuments();
      setFile(null);
    } catch (err) {
      setError('Error processing document. Please try again.');
      setProcessing(false);
      setProcessComplete(false);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await axios.delete(`${API_BASE_URL}${API_ENDPOINTS.DELETE_DOCUMENT(docId)}`);
      setDocuments(documents.filter(doc => doc.id !== docId));
    } catch (err) {
      setError('Error deleting document');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Title Bar */}
      <div className="bg-red-900 text-white">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <FileText className="w-10 h-10 text-red-400" />
            <div className="flex flex-col">
              <h1 className="text-3xl font-['Poppins'] font-bold bg-gradient-to-r from-red-300 to-red-100 bg-clip-text text-transparent">
                Document Manager
              </h1>
              <p className="text-red-200 text-sm">
                Upload and Manage Your Documents
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {/* Upload Section */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-800 flex items-center">
              <FileUp className="w-6 h-6 mr-2 text-red-600" strokeWidth={1.5} />
              Upload New Document
            </h2>
            {processComplete && (
              <div className="flex items-center text-emerald-600 bg-emerald-50 px-4 py-2 rounded-full animate-fadeIn">
                <CheckCircle2 className="w-5 h-5 mr-2" />
                <span className="font-medium">Upload Complete!</span>
              </div>
            )}
          </div>

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
              ${isDragActive ? 'border-red-400 bg-red-50' : 'border-gray-300 hover:border-red-300'}`}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center">
              <Upload className={`w-12 h-12 mb-4 ${isDragActive ? 'text-red-400' : 'text-gray-400'}`} />
              <p className="text-lg font-medium text-gray-700 mb-1">
                Drop your PDF/Image document here
              </p>
              <p className="text-sm text-gray-500">
                or click to browse from your computer
              </p>
              <p className="text-xs text-gray-400 mt-2">
                Supported formats: PDF, JPG, JPEG, PNG
              </p>
            </div>
          </div>

          <button
            onClick={handleFileUpload}
            disabled={!file || processing}
            className={`mt-6 w-full py-3 px-4 rounded-lg font-medium text-white flex items-center justify-center space-x-2
              transform transition-all duration-200
              ${!file || processing
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700 hover:shadow-md active:scale-98'}`}
          >
            {processing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Processing Document...</span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                <span>Upload Document</span>
              </>
            )}
          </button>
        </div>

        {/* Documents List */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-800 flex items-center mb-6">
            <FileText className="w-6 h-6 mr-2 text-red-600" />
            Your Documents
          </h2>

          <div className="space-y-4">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:border-red-200 transition-colors duration-200"
              >
                <div className="flex flex-col flex-grow mr-4">
                  <span className="text-gray-900 font-medium">{doc.name}</span>
                  <span className="text-xs text-gray-500 mt-1">
                    Collection: {doc.original_name}
                  </span>
                </div>
                <div className="flex items-center space-x-3 flex-shrink-0">
                  <button
                    onClick={() => navigate(`/analyze/${doc.id}`)}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors duration-200 flex items-center space-x-2"
                  >
                    <Search className="w-4 h-4" />
                    <span>View Details</span>
                  </button>
                  <button
                    onClick={() => handleDelete(doc.id)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors duration-200"
                    title="Delete Document"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))}
            {documents.length === 0 && (
              <p className="text-center text-gray-500 py-8">
                No documents uploaded yet
              </p>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center text-red-600 animate-fadeIn">
            <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}
      </div>
    </div>
  );
} 