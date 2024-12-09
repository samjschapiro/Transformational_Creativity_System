import React, { useState } from 'react';
import FileUpload from './FileUpload';

function App() {
  const [fileUrl, setFileUrl] = useState(null);
  const [formatType, setFormatType] = useState(null);
  const [formalizationData, setFormalizationData] = useState(null);

  const onFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:3000/upload', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (data.fileUrl) {
      setFileUrl(data.fileUrl);
    }
  };

  const handleFormalize = async (type) => {
    if (!fileUrl) return;
    const response = await fetch('http://localhost:3000/formalize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fileUrl, formatType: type }),
    });
    const data = await response.json();
    setFormalizationData(data);
  };

  return (
    <div style={{margin: '40px'}}>
      <h1>Philosophical Text Formalizer</h1>
      <FileUpload onFileUpload={onFileUpload} />
      {fileUrl && (
        <div>
          <h2>Choose Formalization Type</h2>
          <button onClick={() => handleFormalize('logic')}>Formal Logic</button>
          <button onClick={() => handleFormalize('english')}>English Formalization</button>
        </div>
      )}

      {formalizationData && (
        <div style={{marginTop: '20px'}}>
          <h3>Formalization Results</h3>
          <pre>{JSON.stringify(formalizationData.axioms, null, 2)}</pre>
          {formalizationData.output_pdf_path && (
            <div>
              <a href={`http://localhost:3000/download?path=${encodeURIComponent(formalizationData.output_pdf_path)}`} target="_blank" rel="noopener noreferrer">
                Download Result PDF
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
