import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DocumentList from './pages/DocumentList';
import DocumentAnalysis from './pages/DocumentAnalysis';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<DocumentList />} />
        <Route path="/analyze/:documentId" element={<DocumentAnalysis />} />
      </Routes>
    </Router>
  );
}

export default App;