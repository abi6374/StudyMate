import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Paper,
  Chip,
  Divider,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip,
  IconButton,
  Fade,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent
} from '@mui/material';
import {
  Send as SendIcon,
  Psychology as PsychologyIcon,
  AutoAwesome as AutoAwesomeIcon,
  Science as ScienceIcon,
  Speed as SpeedIcon,
  ExpandMore as ExpandMoreIcon,
  Source as SourceIcon,
  ContentCopy as ContentCopyIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon
} from '@mui/icons-material';

interface Source {
  doc_id: string;
  chunk: string;
  score: number;
}

interface QAResponse {
  query: string;
  answer: string;
  sources: Source[];
  confidence: number;
  timestamp: string;
}

interface ConversationItem {
  id: string;
  question: string;
  response: QAResponse;
  timestamp: string;
}

const QATabProfessional: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [aiModel, setAiModel] = useState('auto');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const handleModelChange = (event: SelectChangeEvent) => {
    setAiModel(event.target.value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/qa/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          question: question.trim(),
          ai_model: aiModel
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: QAResponse = await response.json();
      
      const newItem: ConversationItem = {
        id: Date.now().toString(),
        question: question.trim(),
        response: data,
        timestamp: new Date().toISOString()
      };

      setConversation(prev => [...prev, newItem]);
      setQuestion('');
    } catch (error) {
      console.error('Error asking question:', error);
      setError(error instanceof Error ? error.message : 'Failed to get answer');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  const features = [
    {
      icon: <PsychologyIcon color="primary" />,
      title: "AI-Powered Analysis",
      description: "Advanced semantic understanding using SentenceTransformers"
    },
    {
      icon: <ScienceIcon color="secondary" />,
      title: "FAISS Vector Search",
      description: "Lightning-fast similarity search across your documents"
    },
    {
      icon: <AutoAwesomeIcon color="warning" />,
      title: "IBM Watson LLM",
      description: "Powered by Mixtral-8x7B-Instruct for accurate answers"
    },
    {
      icon: <SpeedIcon color="success" />,
      title: "Real-time Processing",
      description: "Instant responses with contextual source references"
    }
  ];

  const suggestedQuestions = [
    "What are the main concepts covered in this document?",
    "Can you summarize the key findings?",
    "What methodology was used in this study?",
    "What are the conclusions and recommendations?",
    "How does this relate to current industry practices?"
  ];

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Card sx={{ mb: 2, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <CardContent>
          <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, fontWeight: 'bold' }}>
            <PsychologyIcon sx={{ fontSize: 40 }} />
            Professional Q&A Assistant
          </Typography>
          <Typography variant="h6" sx={{ mb: 2, opacity: 0.9 }}>
            Ask natural language questions about your uploaded documents
          </Typography>
          <Typography variant="body1" sx={{ opacity: 0.8 }}>
            Our advanced AI system uses semantic search with TF-IDF and Google Gemini to provide accurate, contextual answers.
          </Typography>
        </CardContent>
      </Card>

      {/* Conversation Area */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
        {conversation.length === 0 ? (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🚀 Ready to explore your documents!
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Start by asking a question about your uploaded PDFs. Here are some suggestions:
              </Typography>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                {suggestedQuestions.map((suggestion, index) => (
                  <Chip
                    key={index}
                    label={suggestion}
                    variant="outlined"
                    onClick={() => setQuestion(suggestion)}
                    sx={{ cursor: 'pointer' }}
                  />
                ))}
              </Box>
            </CardContent>
          </Card>
        ) : (
          conversation.map((item) => (
            <Fade in={true} key={item.id}>
              <Card sx={{ mb: 2 }}>
                <CardContent>
                  {/* Question */}
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <PsychologyIcon color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6" color="primary">
                      Your Question
                    </Typography>
                  </Box>
                  <Typography variant="body1" sx={{ mb: 2, fontStyle: 'italic' }}>
                    "{item.question}"
                  </Typography>

                  <Divider sx={{ my: 2 }} />

                  {/* Answer */}
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <AutoAwesomeIcon color="secondary" sx={{ mr: 1 }} />
                    <Typography variant="h6" color="secondary">
                      AI Answer
                    </Typography>
                    <Chip
                      label={getConfidenceLabel(item.response.confidence)}
                      color={getConfidenceColor(item.response.confidence)}
                      size="small"
                      sx={{ ml: 2 }}
                    />
                    <Tooltip title="Copy answer">
                      <IconButton 
                        size="small" 
                        onClick={() => copyToClipboard(item.response.answer)}
                        sx={{ ml: 1 }}
                      >
                        <ContentCopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  
                  <Typography variant="body1" sx={{ mb: 2, lineHeight: 1.6 }}>
                    {item.response.answer}
                  </Typography>

                  {/* Sources */}
                  {item.response.sources && item.response.sources.length > 0 && (
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <SourceIcon sx={{ mr: 1 }} />
                          <Typography variant="subtitle1">
                            Sources ({item.response.sources.length})
                          </Typography>
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <List>
                          {item.response.sources.map((source, index) => (
                            <ListItem key={index} divider>
                              <ListItemIcon>
                                <Chip 
                                  label={`${(source.score * 100).toFixed(1)}%`}
                                  color="primary"
                                  size="small"
                                />
                              </ListItemIcon>
                              <ListItemText
                                primary={`Document: ${source.doc_id}`}
                                secondary={
                                  <Typography variant="body2" color="text.secondary">
                                    {source.chunk}
                                  </Typography>
                                }
                              />
                            </ListItem>
                          ))}
                        </List>
                      </AccordionDetails>
                    </Accordion>
                  )}

                  {/* Feedback */}
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ flexGrow: 1 }}>
                      {new Date(item.response.timestamp).toLocaleString()}
                    </Typography>
                    <Tooltip title="Helpful">
                      <IconButton size="small" color="success">
                        <ThumbUpIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Not helpful">
                      <IconButton size="small" color="error">
                        <ThumbDownIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </CardContent>
              </Card>
            </Fade>
          ))
        )}
        <div ref={messagesEndRef} />
      </Box>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* AI Model Selection */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <AutoAwesomeIcon color="primary" />
            <Typography variant="h6">AI Model Selection</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ ml: 'auto' }}>
              Current: <strong>{aiModel.toUpperCase()}</strong>
            </Typography>
          </Box>
          
          {/* Simplified dropdown for testing */}
          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            <Button
              variant={aiModel === 'auto' ? 'contained' : 'outlined'}
              onClick={() => setAiModel('auto')}
              startIcon={<AutoAwesomeIcon />}
              size="small"
            >
              Auto
            </Button>
            <Button
              variant={aiModel === 'gemini' ? 'contained' : 'outlined'}
              onClick={() => setAiModel('gemini')}
              startIcon={<PsychologyIcon />}
              size="small"
            >
              Gemini
            </Button>
            <Button
              variant={aiModel === 'local' ? 'contained' : 'outlined'}
              onClick={() => setAiModel('local')}
              startIcon={<ScienceIcon />}
              size="small"
            >
              Local
            </Button>
          </Box>
          
          <Typography variant="caption" color="text.secondary">
            {aiModel === 'auto' && 'Uses Gemini AI if available, falls back to local model'}
            {aiModel === 'gemini' && 'Advanced AI with superior understanding (requires API key)'}
            {aiModel === 'local' && 'Fast, privacy-focused local processing'}
          </Typography>
        </CardContent>
      </Card>

      {/* Question Input */}
      <Card sx={{ mb: 2, border: '2px solid', borderColor: 'primary.main', borderRadius: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SendIcon color="primary" />
            Ask Your Question
          </Typography>
          <form onSubmit={handleSubmit}>
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
              <TextField
                fullWidth
                multiline
                minRows={2}
                maxRows={6}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Type your question here... (e.g., 'What are the main concepts covered in this document?')"
                disabled={loading}
                variant="outlined"
                helperText="Press Ctrl+Enter to send, or use the Send button"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    fontSize: '1.1rem',
                    '&:hover fieldset': {
                      borderColor: 'primary.main',
                    },
                  },
                }}
              />
              <Button
                type="submit"
                variant="contained"
                disabled={!question.trim() || loading}
                sx={{ 
                  minWidth: 120, 
                  height: 60,
                  borderRadius: 2,
                  fontSize: '1rem',
                  fontWeight: 'bold'
                }}
                startIcon={loading ? null : <SendIcon />}
              >
                {loading ? (
                  <CircularProgress size={24} color="inherit" />
                ) : (
                  'Send'
                )}
              </Button>
            </Box>
          </form>
          
          {/* Suggested Questions */}
          {conversation.length === 0 && (
            <Box sx={{ mt: 3, pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                💡 Try these sample questions:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {suggestedQuestions.map((suggestion, index) => (
                  <Chip
                    key={index}
                    label={suggestion}
                    variant="outlined"
                    onClick={() => setQuestion(suggestion)}
                    sx={{ 
                      cursor: 'pointer',
                      '&:hover': {
                        backgroundColor: 'primary.light',
                        color: 'white'
                      }
                    }}
                  />
                ))}
              </Box>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default QATabProfessional;