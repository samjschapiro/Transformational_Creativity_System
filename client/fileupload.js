import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';

const FileUpload = ({ onFileUpload }) => {
  const [uploadedFileName, setUploadedFileName] = useState(null);

  const onDrop = (acceptedFiles) => {
    const uploadedFile = acceptedFiles[0];
    setUploadedFileName(uploadedFile.name);
    onFileUpload(uploadedFile);
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  return (
    <div style={{margin: '20px'}}>
      <div {...getRootProps()} style={dropzoneStyles}>
        <input {...getInputProps()} />
        <p>Drag & drop PDF or EPUB here, or click to select files</p>
      </div>
      {uploadedFileName && <p>File '{uploadedFileName}' uploaded successfully!</p>}
    </div>
  );
};

const dropzoneStyles = {
  border: '2px dashed #cccccc',
  borderRadius: '4px',
  padding: '20px',
  textAlign: 'center',
  cursor: 'pointer',
};

export default FileUpload;
