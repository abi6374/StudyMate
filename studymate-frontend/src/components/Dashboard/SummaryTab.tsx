import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
} from '@mui/material';
import {
  Summarize,
  ExpandMore,
  AutoAwesome,
  Download,
  PictureAsPdf,
} from '@mui/icons-material';
import { toast } from 'react-toastify';

interface SummaryTabProps {
  uploadedFiles: string[];
}

const SummaryTab: React.FC<SummaryTabProps> = ({ uploadedFiles }) => {
  const [summaries, setSummaries] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [summaryLength, setSummaryLength] = useState(3);

  const generateSummary = async (fileName: string) => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      const newSummary = {
        fileName,
        summary: `This is a comprehensive summary of ${fileName}. The document covers key concepts including theoretical frameworks, practical applications, and research methodologies. Key findings include significant insights into the subject matter with detailed analysis and conclusions. The content is organized into multiple sections covering background information, main arguments, supporting evidence, and future implications. This summary provides a condensed overview while maintaining the essential information and context from the original document.`,
        keyPoints: [
          'Main theoretical framework and concepts',
          'Research methodology and approach',
          'Key findings and results',
          'Practical applications and implications',
          'Conclusions and future work'
        ],
        length: summaryLength,
        timestamp: new Date().toLocaleString(),
      };

      setSummaries(prev => [newSummary, ...prev]);
      toast.success('Summary generated successfully!');
    } catch (error) {
      toast.error('Failed to generate summary');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        Document Summaries
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Generate AI-powered summaries of your uploaded documents
      </Typography>

      {uploadedFiles.length === 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Please upload some PDF documents first to generate summaries.
        </Alert>
      )}

      {/* Summary Controls */}
      {uploadedFiles.length > 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
            Generate New Summary
          </Typography>
          
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Summary Length
            </Typography>
            <Slider
              value={summaryLength}
              onChange={(_, value) => setSummaryLength(value as number)}
              min={1}
              max={5}
              marks={[
                { value: 1, label: 'Brief' },
                { value: 3, label: 'Moderate' },
                { value: 5, label: 'Detailed' },
              ]}
              sx={{ mb: 2 }}
            />
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Select document to summarize:
          </Typography>
          
          <List sx={{ mb: 2 }}>
            {uploadedFiles.map((file, index) => (
              <ListItem
                key={index}
                sx={{
                  border: '1px solid',
                  borderColor: 'grey.200',
                  borderRadius: 1,
                  mb: 1,
                  bgcolor: 'grey.50',
                }}
              >
                <ListItemIcon>
                  <PictureAsPdf color="error" />
                </ListItemIcon>
                <ListItemText
                  primary={file}
                  secondary="Ready for summarization"
                />
                <Button
                  variant="contained"
                  size="small"
                  startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <AutoAwesome />}
                  onClick={() => generateSummary(file)}
                  disabled={loading}
                >
                  {loading ? 'Generating...' : 'Summarize'}
                </Button>
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Generated Summaries */}
      {summaries.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Generated Summaries ({summaries.length})
          </Typography>
          
          {summaries.map((summary, index) => (
            <Accordion key={index} defaultExpanded={index === 0}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                  <Summarize color="primary" />
                  <Typography sx={{ fontWeight: 500, flexGrow: 1 }}>
                    {summary.fileName}
                  </Typography>
                  <Chip
                    label={`${summary.length === 1 ? 'Brief' : summary.length === 3 ? 'Moderate' : 'Detailed'}`}
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                  <Typography variant="caption" color="text.secondary">
                    {summary.timestamp}
                  </Typography>
                </Box>
              </AccordionSummary>
              
              <AccordionDetails>
                <Box>
                  <Typography variant="body1" sx={{ mb: 3, lineHeight: 1.7 }}>
                    {summary.summary}
                  </Typography>
                  
                  <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
                    Key Points:
                  </Typography>
                  <List dense>
                    {summary.keyPoints.map((point: string, pointIndex: number) => (
                      <ListItem key={pointIndex} sx={{ py: 0.5 }}>
                        <ListItemText 
                          primary={`• ${point}`}
                          sx={{ '& .MuiListItemText-primary': { fontSize: '0.875rem' } }}
                        />
                      </ListItem>
                    ))}
                  </List>
                  
                  <Box sx={{ mt: 3, display: 'flex', gap: 1 }}>
                    <Button
                      size="small"
                      startIcon={<Download />}
                      variant="outlined"
                    >
                      Export Summary
                    </Button>
                  </Box>
                </Box>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}

      {summaries.length === 0 && uploadedFiles.length > 0 && (
        <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.50' }}>
          <Summarize sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="body1" color="text.secondary">
            No summaries generated yet. Select a document above to create your first summary!
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default SummaryTab;
