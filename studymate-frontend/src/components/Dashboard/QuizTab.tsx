import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Card,
  CardContent,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  CircularProgress,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  Quiz,
  CheckCircle,
  Cancel,
  PictureAsPdf,
  AutoAwesome,
  PlayArrow,
  Refresh,
} from '@mui/icons-material';
import { toast } from 'react-toastify';

interface QuizTabProps {
  uploadedFiles: string[];
}

interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
}

interface QuizResult {
  questionId: string;
  selectedAnswer: number;
  isCorrect: boolean;
}

const QuizTab: React.FC<QuizTabProps> = ({ uploadedFiles }) => {
  const [quiz, setQuiz] = useState<QuizQuestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<{ [key: string]: number }>({});
  const [results, setResults] = useState<QuizResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [quizStarted, setQuizStarted] = useState(false);

  const generateQuiz = async (fileName: string) => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      const newQuiz: QuizQuestion[] = [
        {
          id: '1',
          question: 'What is the primary focus of the theoretical framework discussed?',
          options: [
            'Cognitive load management',
            'Social interaction patterns',
            'Memory retention techniques',
            'Learning style adaptation'
          ],
          correctAnswer: 0,
          explanation: 'The document primarily focuses on cognitive load theory and how it affects learning processes.',
          difficulty: 'Medium',
        },
        {
          id: '2',
          question: 'Which research methodology was predominantly used in the study?',
          options: [
            'Purely quantitative approach',
            'Mixed-methods research',
            'Qualitative interviews only',
            'Literature review'
          ],
          correctAnswer: 1,
          explanation: 'The study employed a mixed-methods approach combining both quantitative and qualitative data collection.',
          difficulty: 'Easy',
        },
        {
          id: '3',
          question: 'What was the most significant finding regarding student engagement?',
          options: [
            'Technology integration had no impact',
            'Traditional methods were more effective',
            'Active learning strategies improved outcomes',
            'Group size did not matter'
          ],
          correctAnswer: 2,
          explanation: 'The research showed that active learning strategies significantly improved student engagement and outcomes.',
          difficulty: 'Hard',
        },
        {
          id: '4',
          question: 'According to the document, what is essential for personalized learning?',
          options: [
            'Standardized curriculum',
            'Individual learning pace adaptation',
            'Uniform assessment methods',
            'Fixed learning schedules'
          ],
          correctAnswer: 1,
          explanation: 'The document emphasizes the importance of adapting to individual learning paces for effective personalized learning.',
          difficulty: 'Medium',
        },
        {
          id: '5',
          question: 'What future research direction is suggested?',
          options: [
            'Reducing technology use',
            'Focusing only on traditional methods',
            'Investigating AI-powered learning systems',
            'Eliminating collaborative learning'
          ],
          correctAnswer: 2,
          explanation: 'The document suggests future research should explore AI-powered personalized learning systems.',
          difficulty: 'Hard',
        },
      ];

      setQuiz(newQuiz);
      setCurrentQuestion(0);
      setAnswers({});
      setResults([]);
      setShowResults(false);
      setQuizStarted(false);
      toast.success(`Generated quiz with ${newQuiz.length} questions from ${fileName}!`);
    } catch (error) {
      toast.error('Failed to generate quiz');
    } finally {
      setLoading(false);
    }
  };

  const startQuiz = () => {
    setQuizStarted(true);
    setCurrentQuestion(0);
    setAnswers({});
    setResults([]);
    setShowResults(false);
  };

  const handleAnswerChange = (questionId: string, answerIndex: number) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: answerIndex
    }));
  };

  const nextQuestion = () => {
    if (currentQuestion < quiz.length - 1) {
      setCurrentQuestion(prev => prev + 1);
    }
  };

  const previousQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1);
    }
  };

  const submitQuiz = () => {
    const quizResults: QuizResult[] = quiz.map(question => ({
      questionId: question.id,
      selectedAnswer: answers[question.id] ?? -1,
      isCorrect: answers[question.id] === question.correctAnswer,
    }));

    setResults(quizResults);
    setShowResults(true);
    
    const correctAnswers = quizResults.filter(result => result.isCorrect).length;
    const percentage = Math.round((correctAnswers / quiz.length) * 100);
    toast.success(`Quiz completed! Score: ${correctAnswers}/${quiz.length} (${percentage}%)`);
  };

  const resetQuiz = () => {
    setQuizStarted(false);
    setCurrentQuestion(0);
    setAnswers({});
    setResults([]);
    setShowResults(false);
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Easy': return 'success';
      case 'Medium': return 'warning';
      case 'Hard': return 'error';
      default: return 'primary';
    }
  };

  const getScoreColor = (percentage: number) => {
    if (percentage >= 80) return 'success';
    if (percentage >= 60) return 'warning';
    return 'error';
  };

  if (showResults) {
    const correctAnswers = results.filter(result => result.isCorrect).length;
    const percentage = Math.round((correctAnswers / quiz.length) * 100);

    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
          Quiz Results
        </Typography>
        
        <Paper sx={{ p: 3, mb: 3, textAlign: 'center' }}>
          <Typography variant="h4" sx={{ fontWeight: 600, mb: 2 }}>
            {percentage}%
          </Typography>
          <Typography variant="h6" gutterBottom>
            {correctAnswers} out of {quiz.length} correct
          </Typography>
          <LinearProgress
            variant="determinate"
            value={percentage}
            color={getScoreColor(percentage) as any}
            sx={{ height: 8, borderRadius: 4, mb: 2 }}
          />
          <Button
            startIcon={<Refresh />}
            onClick={resetQuiz}
            variant="contained"
          >
            Try Again
          </Button>
        </Paper>

        <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
          Review Answers
        </Typography>

        {quiz.map((question, index) => {
          const result = results.find(r => r.questionId === question.id);
          const isCorrect = result?.isCorrect ?? false;
          const selectedAnswer = result?.selectedAnswer ?? -1;

          return (
            <Card key={question.id} sx={{ mb: 2 }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  {isCorrect ? (
                    <CheckCircle color="success" />
                  ) : (
                    <Cancel color="error" />
                  )}
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    Question {index + 1}
                  </Typography>
                  <Chip
                    label={question.difficulty}
                    size="small"
                    color={getDifficultyColor(question.difficulty) as any}
                    variant="outlined"
                  />
                </Box>

                <Typography variant="body1" sx={{ mb: 2 }}>
                  {question.question}
                </Typography>

                <Box sx={{ mb: 2 }}>
                  {question.options.map((option, optionIndex) => (
                    <Box
                      key={optionIndex}
                      sx={{
                        p: 1,
                        borderRadius: 1,
                        mb: 1,
                        bgcolor: 
                          optionIndex === question.correctAnswer 
                            ? 'success.100' 
                            : optionIndex === selectedAnswer && !isCorrect
                            ? 'error.100'
                            : 'transparent',
                        border: '1px solid',
                        borderColor:
                          optionIndex === question.correctAnswer 
                            ? 'success.main' 
                            : optionIndex === selectedAnswer && !isCorrect
                            ? 'error.main'
                            : 'grey.300',
                      }}
                    >
                      <Typography variant="body2">
                        {String.fromCharCode(65 + optionIndex)}. {option}
                        {optionIndex === question.correctAnswer && ' ✓'}
                        {optionIndex === selectedAnswer && !isCorrect && ' ✗'}
                      </Typography>
                    </Box>
                  ))}
                </Box>

                <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Typography variant="body2" color="text.secondary">
                    <strong>Explanation:</strong> {question.explanation}
                  </Typography>
                </Paper>
              </CardContent>
            </Card>
          );
        })}
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        Quiz Generation
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Create self-assessment quizzes from your study materials
      </Typography>

      {uploadedFiles.length === 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Please upload some PDF documents first to generate quizzes.
        </Alert>
      )}

      {/* Generate Quiz */}
      {uploadedFiles.length > 0 && !quizStarted && quiz.length === 0 && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
            Generate Quiz
          </Typography>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Select a document to generate quiz questions from:
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
                  secondary="Ready for quiz generation"
                />
                <Button
                  variant="contained"
                  size="small"
                  startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <AutoAwesome />}
                  onClick={() => generateQuiz(file)}
                  disabled={loading}
                >
                  {loading ? 'Generating...' : 'Generate Quiz'}
                </Button>
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Quiz Ready */}
      {quiz.length > 0 && !quizStarted && (
        <Paper sx={{ p: 3, mb: 3, textAlign: 'center' }}>
          <Quiz sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Quiz Ready!
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Your quiz has {quiz.length} questions. Ready to test your knowledge?
          </Typography>
          <Button
            variant="contained"
            size="large"
            startIcon={<PlayArrow />}
            onClick={startQuiz}
          >
            Start Quiz
          </Button>
        </Paper>
      )}

      {/* Quiz Interface */}
      {quizStarted && quiz.length > 0 && (
        <Box>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6">
                Question {currentQuestion + 1} of {quiz.length}
              </Typography>
              <Chip
                label={quiz[currentQuestion].difficulty}
                color={getDifficultyColor(quiz[currentQuestion].difficulty) as any}
                variant="outlined"
              />
            </Box>

            <LinearProgress
              variant="determinate"
              value={((currentQuestion + 1) / quiz.length) * 100}
              sx={{ mb: 3, height: 6, borderRadius: 3 }}
            />

            <Typography variant="h6" sx={{ mb: 3 }}>
              {quiz[currentQuestion].question}
            </Typography>

            <FormControl component="fieldset" sx={{ width: '100%' }}>
              <RadioGroup
                value={answers[quiz[currentQuestion].id] ?? ''}
                onChange={(e) => handleAnswerChange(quiz[currentQuestion].id, parseInt(e.target.value))}
              >
                {quiz[currentQuestion].options.map((option, index) => (
                  <FormControlLabel
                    key={index}
                    value={index}
                    control={<Radio />}
                    label={`${String.fromCharCode(65 + index)}. ${option}`}
                    sx={{
                      mb: 1,
                      p: 1,
                      borderRadius: 1,
                      '&:hover': {
                        bgcolor: 'grey.50',
                      },
                    }}
                  />
                ))}
              </RadioGroup>
            </FormControl>

            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
              <Button
                onClick={previousQuestion}
                disabled={currentQuestion === 0}
                variant="outlined"
              >
                Previous
              </Button>

              <Box sx={{ display: 'flex', gap: 2 }}>
                {currentQuestion === quiz.length - 1 ? (
                  <Button
                    onClick={submitQuiz}
                    variant="contained"
                    color="success"
                    disabled={Object.keys(answers).length !== quiz.length}
                  >
                    Submit Quiz
                  </Button>
                ) : (
                  <Button
                    onClick={nextQuestion}
                    variant="contained"
                    disabled={currentQuestion === quiz.length - 1}
                  >
                    Next
                  </Button>
                )}
              </Box>
            </Box>
          </Paper>

          {/* Progress Overview */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Progress Overview
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {quiz.map((_, index) => (
                <Chip
                  key={index}
                  label={index + 1}
                  size="small"
                  color={answers[quiz[index].id] !== undefined ? 'primary' : 'default'}
                  variant={index === currentQuestion ? 'filled' : 'outlined'}
                  onClick={() => setCurrentQuestion(index)}
                  clickable
                />
              ))}
            </Box>
          </Paper>
        </Box>
      )}

      {quiz.length === 0 && uploadedFiles.length > 0 && !loading && (
        <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.50' }}>
          <Quiz sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="body1" color="text.secondary">
            No quiz generated yet. Select a document above to create your first quiz!
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default QuizTab;
