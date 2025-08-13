import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Card,
  CardContent,
  CardActions,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  CircularProgress,
  Alert,
  Fade,
} from '@mui/material';
import {
  AutoStories,
  Flip,
  Delete,
  Shuffle,
  PictureAsPdf,
  AutoAwesome,
} from '@mui/icons-material';
import { toast } from 'react-toastify';

interface FlashcardsTabProps {
  uploadedFiles: string[];
}

interface Flashcard {
  id: string;
  question: string;
  answer: string;
  source: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
}

const FlashcardsTab: React.FC<FlashcardsTabProps> = ({ uploadedFiles }) => {
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [loading, setLoading] = useState(false);
  const [flipped, setFlipped] = useState<{ [key: string]: boolean }>({});

  const generateFlashcards = async (fileName: string) => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      const newFlashcards: Flashcard[] = [
        {
          id: '1',
          question: 'What are the main theoretical frameworks discussed in this document?',
          answer: 'The document covers several key theoretical frameworks including cognitive load theory, constructivist learning principles, and social learning theory. These frameworks provide the foundation for understanding how students process and retain information.',
          source: fileName,
          difficulty: 'Medium',
        },
        {
          id: '2',
          question: 'What methodology was used in the research presented?',
          answer: 'The research employed a mixed-methods approach combining quantitative surveys with qualitative interviews. This methodology allowed for comprehensive data collection and analysis.',
          source: fileName,
          difficulty: 'Hard',
        },
        {
          id: '3',
          question: 'What are the key findings mentioned in the study?',
          answer: 'Key findings include improved learning outcomes when using active learning strategies, increased student engagement through technology integration, and the importance of personalized learning approaches.',
          source: fileName,
          difficulty: 'Easy',
        },
        {
          id: '4',
          question: 'What practical applications are suggested?',
          answer: 'The document suggests implementing spaced repetition techniques, using multimedia content for better retention, and creating collaborative learning environments to enhance educational outcomes.',
          source: fileName,
          difficulty: 'Medium',
        },
        {
          id: '5',
          question: 'What are the future research directions mentioned?',
          answer: 'Future research should focus on longitudinal studies of learning outcomes, investigation of AI-powered personalized learning systems, and exploration of virtual reality applications in education.',
          source: fileName,
          difficulty: 'Hard',
        },
      ];

      setFlashcards(prev => [...prev, ...newFlashcards]);
      toast.success(`Generated ${newFlashcards.length} flashcards from ${fileName}!`);
    } catch (error) {
      toast.error('Failed to generate flashcards');
    } finally {
      setLoading(false);
    }
  };

  const flipCard = (cardId: string) => {
    setFlipped(prev => ({
      ...prev,
      [cardId]: !prev[cardId]
    }));
  };

  const shuffleCards = () => {
    const shuffled = [...flashcards].sort(() => Math.random() - 0.5);
    setFlashcards(shuffled);
    toast.info('Flashcards shuffled!');
  };

  const deleteCard = (cardId: string) => {
    setFlashcards(prev => prev.filter(card => card.id !== cardId));
    toast.info('Flashcard deleted');
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Easy': return 'success';
      case 'Medium': return 'warning';
      case 'Hard': return 'error';
      default: return 'primary';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        AI Flashcards
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Auto-generate flashcards from key concepts in your documents
      </Typography>

      {uploadedFiles.length === 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Please upload some PDF documents first to generate flashcards.
        </Alert>
      )}

      {/* Generate Flashcards */}
      {uploadedFiles.length > 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
            Generate Flashcards
          </Typography>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Select a document to generate flashcards from:
          </Typography>
          
          <List>
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
                  secondary="Ready for flashcard generation"
                />
                <Button
                  variant="contained"
                  size="small"
                  startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <AutoAwesome />}
                  onClick={() => generateFlashcards(file)}
                  disabled={loading}
                >
                  {loading ? 'Generating...' : 'Generate Cards'}
                </Button>
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Flashcard Controls */}
      {flashcards.length > 0 && (
        <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Study Cards ({flashcards.length})
          </Typography>
          <Button
            startIcon={<Shuffle />}
            onClick={shuffleCards}
            variant="outlined"
            size="small"
          >
            Shuffle
          </Button>
        </Box>
      )}

      {/* Flashcards Grid */}
      {flashcards.length > 0 && (
        <Box 
          sx={{ 
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: 3,
          }}
        >
          {flashcards.map((card) => (
            <Card
              key={card.id}
              sx={{
                height: 280,
                cursor: 'pointer',
                transition: 'transform 0.2s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4,
                },
              }}
              onClick={() => flipCard(card.id)}
            >
                <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Chip
                      label={card.difficulty}
                      size="small"
                      color={getDifficultyColor(card.difficulty) as any}
                      variant="outlined"
                    />
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        flipCard(card.id);
                      }}
                    >
                      <Flip />
                    </IconButton>
                  </Box>
                  
                  <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Fade in={!flipped[card.id]} timeout={300}>
                      <Box sx={{ display: flipped[card.id] ? 'none' : 'block', textAlign: 'center' }}>
                        <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                          Question
                        </Typography>
                        <Typography variant="body1">
                          {card.question}
                        </Typography>
                      </Box>
                    </Fade>
                    
                    <Fade in={flipped[card.id]} timeout={300}>
                      <Box sx={{ display: !flipped[card.id] ? 'none' : 'block', textAlign: 'center' }}>
                        <Typography variant="h6" sx={{ fontWeight: 600, mb: 1, color: 'primary.main' }}>
                          Answer
                        </Typography>
                        <Typography variant="body2" sx={{ lineHeight: 1.5 }}>
                          {card.answer}
                        </Typography>
                      </Box>
                    </Fade>
                  </Box>
                  
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 2 }}>
                    Source: {card.source}
                  </Typography>
                </CardContent>
                
                <CardActions sx={{ justifyContent: 'flex-end' }}>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteCard(card.id);
                    }}
                  >
                    <Delete />
                  </IconButton>
                </CardActions>
              </Card>
          ))}
        </Box>
      )}

      {flashcards.length === 0 && uploadedFiles.length > 0 && (
        <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.50' }}>
          <AutoStories sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="body1" color="text.secondary">
            No flashcards generated yet. Select a document above to create your first set of flashcards!
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default FlashcardsTab;
