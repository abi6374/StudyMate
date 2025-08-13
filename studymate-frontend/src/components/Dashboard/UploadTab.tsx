import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  Paper,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
} from '@mui/material';
import {
  CloudUpload,
  PictureAsPdf,
  Delete,
  CheckCircle,
} from '@mui/icons-material';
import { toast } from 'react-toastify';

interface UploadTabProps {
  uploadedFiles: string[];
  setUploadedFiles: React.Dispatch<React.SetStateAction<string[]>>;
}

const UploadTab: React.FC<UploadTabProps> = ({ uploadedFiles, setUploadedFiles }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = async (acceptedFiles: File[]) => {
    const pdfFiles = acceptedFiles.filter(file => file.type === 'application/pdf');
    
    if (pdfFiles.length === 0) {
      toast.error('Please upload only PDF files');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      // Process files
      for (let i = 0; i < pdfFiles.length; i++) {
        const file = pdfFiles[i];
        
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('file', file);

        console.log('Uploading file:', file.name);
        console.log('FormData:', formData);

        // Upload to backend
        const response = await fetch('http://localhost:8000/api/documents/upload', {
          method: 'POST',
          body: formData,
        });

        console.log('Response status:', response.status);
        console.log('Response ok:', response.ok);

        if (!response.ok) {
          const errorData = await response.text();
          console.error('Upload failed:', response.statusText, errorData);
          toast.error(`Upload failed: ${response.statusText}`);
          continue;
        }

        await response.json();
        
        // Update progress
        setUploadProgress(((i + 1) / pdfFiles.length) * 100);

        // Add to uploaded files
        setUploadedFiles(prev => [...prev, file.name]);
        toast.success(`${file.name} uploaded and indexed successfully!`);
      }
    } catch (error) {
      toast.error('Upload failed');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    multiple: true,
  });

  const removeFile = async (fileName: string) => {
    try {
      // Remove file from backend
      const response = await fetch(`http://localhost:8000/api/documents/${encodeURIComponent(fileName)}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        toast.error(`Failed to remove ${fileName} from server`);
        return;
      }
      
      // Remove from local state
      setUploadedFiles(prev => prev.filter(file => file !== fileName));
      toast.info(`${fileName} removed from index`);
    } catch (error) {
      toast.error(`Error removing ${fileName}`);
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Upload & Index PDFs
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Upload your academic PDFs to start asking questions and getting AI-powered answers.
      </Typography>

      {/* Upload Area */}
      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          textAlign: 'center',
          cursor: 'pointer',
          border: isDragActive ? '2px dashed #1976d2' : '2px dashed #ccc',
          bgcolor: isDragActive ? '#f3f9ff' : 'background.paper',
          transition: 'all 0.3s ease',
          mb: 3,
          '&:hover': {
            border: '2px dashed #1976d2',
            bgcolor: '#f8f9fa',
          },
        }}
      >
        <input {...getInputProps()} />
        <CloudUpload sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          {isDragActive
            ? 'Drop the PDF files here...'
            : 'Drag & drop PDF files here, or click to select files'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Supports multiple PDF files
        </Typography>
        
        {uploading && (
          <Box sx={{ mt: 3, maxWidth: 400, mx: 'auto' }}>
            <LinearProgress variant="determinate" value={uploadProgress} />
            <Typography variant="body2" sx={{ mt: 1 }}>
              Processing... {uploadProgress}%
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Upload Instructions */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>Tips:</strong> For best results, upload clear, text-based PDFs. 
          Scanned documents may have reduced accuracy. Each PDF will be processed 
          and indexed for semantic search.
        </Typography>
      </Alert>

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <Paper sx={{ mt: 3 }}>
          <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="h6">
              Indexed Documents ({uploadedFiles.length})
            </Typography>
          </Box>
          <List>
            {uploadedFiles.map((fileName, index) => (
              <ListItem
                key={index}
                secondaryAction={
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => removeFile(fileName)}
                    color="error"
                  >
                    <Delete />
                  </IconButton>
                }
              >
                <ListItemIcon>
                  <PictureAsPdf color="error" />
                </ListItemIcon>
                <ListItemText
                  primary={fileName}
                  secondary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CheckCircle color="success" sx={{ fontSize: 16 }} />
                      <Typography variant="caption" color="success.main">
                        Indexed and ready for questions
                      </Typography>
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {uploadedFiles.length === 0 && !uploading && (
        <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.50' }}>
          <Typography variant="body1" color="text.secondary">
            No documents uploaded yet. Upload your first PDF to get started!
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default UploadTab;
