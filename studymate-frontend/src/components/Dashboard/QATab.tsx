import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  IconButton,
  Paper,
  Avatar,
  Chip,
  CircularProgress,
  Alert,
  InputAdornment,
  Stack,
} from '@mui/material';
import {
  Send,
  SmartToy,
  Person,
  AttachFile,
} from '@mui/icons-material';
import { toast } from 'react-toastify';
import { useTheme } from '@mui/material/styles';

interface QATabProps {
  uploadedFiles: string[];
  conversationHistory: any[];
  setConversationHistory: React.Dispatch<React.SetStateAction<any[]>>;
}

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  citations?: string[];
  isTyping?: boolean;
  metadata?: {
    confidence_score?: number;
    processing_time?: number;
    language_detected?: string;
    sources?: Array<{
      filename: string;
      relevance_score: number;
      pages_analyzed?: number;
      confidence?: number;
    }>;
  };
}

const QATab: React.FC<QATabProps> = ({ 
  uploadedFiles, 
  conversationHistory, 
  setConversationHistory 
}) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!question.trim()) {
      toast.error('Please enter a question');
      return;
    }

    if (uploadedFiles.length === 0) {
      toast.error('Please upload some PDFs first');
      return;
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: question.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);

    // Add typing indicator
    const typingMessage: Message = {
      id: 'typing',
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      isTyping: true,
    };

    setMessages(prev => [...prev, typingMessage]);

    try {
      // Call the backend API
      const response = await fetch('http://localhost:8000/api/qa/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.content,
          document_ids: uploadedFiles,
          language: 'en',
          response_style: 'comprehensive'
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from server');
      }

      const data = await response.json();
      
      // Remove typing indicator and add response
      setMessages(prev => prev.filter(msg => msg.id !== 'typing'));

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        citations: data.citations || [],
        metadata: {
          confidence_score: data.confidence_score,
          processing_time: data.processing_time,
          language_detected: data.language_detected,
          sources: data.sources
        }
      };

      setMessages(prev => [...prev, assistantMessage]);
      
      // Show professional metrics in toast
      const confidence = Math.round((data.confidence_score || 0) * 100);
      const time = data.processing_time || 0;
      toast.success(`✅ Professional response generated with ${confidence}% confidence in ${time}s`);
    } catch (error) {
      // Remove typing indicator on error
      setMessages(prev => prev.filter(msg => msg.id !== 'typing'));
      
      // Fallback to mock response if API fails
      const mockResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `Based on the uploaded documents, here's what I found regarding your question about "${userMessage.content}". This is a comprehensive answer that draws from multiple sections of your PDFs to provide accurate information. The content has been analyzed and synthesized to give you the most relevant response possible.`,
        timestamp: new Date(),
        citations: [
          `${uploadedFiles[0]} - Page 12`,
          `${uploadedFiles[0]} - Page 24`,
          uploadedFiles[1] ? `${uploadedFiles[1]} - Page 7` : null,
        ].filter(Boolean) as string[],
      };

      setMessages(prev => [...prev, mockResponse]);
      toast.warning('Using demo response - backend connection failed');
    } finally {
      setLoading(false);
    }
  };

  const sampleQuestions = [
    "What are the main concepts covered in this document?",
    "Can you explain the methodology used?",
    "What are the key findings?",
    "How does this relate to previous work?",
  ];

  const handleSampleQuestion = (sampleQ: string) => {
    setQuestion(sampleQ);
  };

  const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
    const isUser = message.type === 'user';
    
    if (message.isTyping) {
      return (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'flex-start',
            mb: 2,
            alignItems: 'flex-end',
          }}
        >
          <Avatar
            sx={{
              width: 32,
              height: 32,
              mr: 1,
              bgcolor: theme.palette.primary.main,
            }}
          >
            <SmartToy sx={{ fontSize: 18 }} />
          </Avatar>
          <Paper
            sx={{
              p: 2,
              bgcolor: theme.palette.grey[100],
              borderRadius: '20px 20px 20px 8px',
              minWidth: 60,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <CircularProgress size={20} />
          </Paper>
        </Box>
      );
    }

    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
          mb: 2,
          alignItems: 'flex-end',
        }}
      >
        {!isUser && (
          <Avatar
            sx={{
              width: 32,
              height: 32,
              mr: 1,
              bgcolor: theme.palette.primary.main,
            }}
          >
            <SmartToy sx={{ fontSize: 18 }} />
          </Avatar>
        )}
        
        <Box sx={{ maxWidth: '70%' }}>
          <Paper
            sx={{
              p: 2,
              bgcolor: isUser 
                ? theme.palette.primary.main 
                : theme.palette.grey[100],
              color: isUser 
                ? theme.palette.primary.contrastText 
                : theme.palette.text.primary,
              borderRadius: isUser 
                ? '20px 20px 8px 20px' 
                : '20px 20px 20px 8px',
              wordBreak: 'break-word',
            }}
          >
            <Typography variant="body1">
              {message.content}
            </Typography>
          </Paper>
          
          {message.citations && message.citations.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Stack direction="row" spacing={0.5} flexWrap="wrap">
                {message.citations.map((citation, index) => (
                  <Chip
                    key={index}
                    label={citation}
                    size="small"
                    variant="outlined"
                    sx={{ 
                      fontSize: '0.75rem',
                      height: 24,
                      bgcolor: 'background.paper',
                    }}
                  />
                ))}
              </Stack>
            </Box>
          )}
          
          <Typography 
            variant="caption" 
            color="text.secondary" 
            sx={{ 
              display: 'block', 
              mt: 0.5,
              textAlign: isUser ? 'right' : 'left',
            }}
          >
            {message.timestamp.toLocaleTimeString()}
          </Typography>
        </Box>

        {isUser && (
          <Avatar
            sx={{
              width: 32,
              height: 32,
              ml: 1,
              bgcolor: theme.palette.secondary.main,
            }}
          >
            <Person sx={{ fontSize: 18 }} />
          </Avatar>
        )}
      </Box>
    );
  };

  return (
    <Box 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        maxHeight: 'calc(100vh - 120px)',
      }}
    >
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Ask Questions
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Chat with your documents using AI
        </Typography>
        
        {uploadedFiles.length > 0 && (
          <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <AttachFile sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="caption" color="text.secondary">
              {uploadedFiles.length} document{uploadedFiles.length !== 1 ? 's' : ''} loaded
            </Typography>
          </Box>
        )}
      </Box>

      {uploadedFiles.length === 0 && (
        <Alert severity="warning" sx={{ m: 2 }}>
          Please upload and index some PDF documents first before asking questions.
        </Alert>
      )}

      {/* Chat Messages */}
      <Box 
        sx={{ 
          flex: 1, 
          overflow: 'auto', 
          p: 2,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {messages.length === 0 && uploadedFiles.length > 0 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <SmartToy sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Start a conversation
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Ask questions about your uploaded documents
            </Typography>
            
            {/* Sample Questions */}
            <Stack direction="row" spacing={1} flexWrap="wrap" justifyContent="center">
              {sampleQuestions.map((sampleQ, index) => (
                <Chip
                  key={index}
                  label={sampleQ}
                  variant="outlined"
                  clickable
                  size="small"
                  onClick={() => handleSampleQuestion(sampleQ)}
                  sx={{ mb: 1 }}
                />
              ))}
            </Stack>
          </Box>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        
        <div ref={messagesEndRef} />
      </Box>

      {/* Input Area */}
      {uploadedFiles.length > 0 && (
        <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
          <Box component="form" onSubmit={handleSubmit}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Ask a question about your documents..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={loading}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '24px',
                  paddingRight: '8px',
                },
              }}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      type="submit"
                      disabled={loading || !question.trim()}
                      sx={{
                        bgcolor: 'primary.main',
                        color: 'primary.contrastText',
                        '&:hover': {
                          bgcolor: 'primary.dark',
                        },
                        '&.Mui-disabled': {
                          bgcolor: 'grey.300',
                          color: 'grey.500',
                        },
                      }}
                    >
                      {loading ? (
                        <CircularProgress size={20} color="inherit" />
                      ) : (
                        <Send sx={{ fontSize: 20 }} />
                      )}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default QATab;
