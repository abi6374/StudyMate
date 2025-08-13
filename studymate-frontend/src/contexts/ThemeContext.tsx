import React, { createContext, useContext, useState, useEffect } from 'react';
import { createTheme, Theme } from '@mui/material/styles';

interface ThemeContextType {
  darkMode: boolean;
  toggleTheme: () => void;
  theme: Theme;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

const createAppTheme = (mode: 'light' | 'dark') => createTheme({
  palette: {
    mode,
    primary: {
      main: mode === 'dark' ? '#6366f1' : '#4f46e5',
      dark: mode === 'dark' ? '#4338ca' : '#3730a3',
      light: mode === 'dark' ? '#818cf8' : '#6366f1',
    },
    secondary: {
      main: mode === 'dark' ? '#ec4899' : '#dc2626',
    },
    background: {
      default: mode === 'dark' ? '#0f0f23' : '#fafbfc',
      paper: mode === 'dark' ? '#1a1b3a' : '#ffffff',
    },
    text: {
      primary: mode === 'dark' ? '#f1f5f9' : '#1e293b',
      secondary: mode === 'dark' ? '#94a3b8' : '#64748b',
    },
    divider: mode === 'dark' ? '#334155' : '#e2e8f0',
    action: {
      hover: mode === 'dark' ? '#1e293b20' : '#f1f5f920',
    },
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", "Roboto", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 800,
      letterSpacing: '-0.02em',
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 700,
      letterSpacing: '-0.02em',
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 700,
      letterSpacing: '-0.01em',
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 700,
      letterSpacing: '-0.01em',
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h6: {
      fontSize: '1.125rem',
      fontWeight: 600,
    },
    body1: {
      lineHeight: 1.7,
      fontSize: '1rem',
    },
    body2: {
      lineHeight: 1.6,
      fontSize: '0.875rem',
    },
  },
  shape: {
    borderRadius: 16,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: '12px',
          padding: '10px 24px',
          fontSize: '0.95rem',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: mode === 'dark' 
              ? '0 4px 20px rgba(99, 102, 241, 0.3)' 
              : '0 4px 20px rgba(79, 70, 229, 0.2)',
          },
        },
        contained: {
          background: mode === 'dark' 
            ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' 
            : 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
          '&:hover': {
            background: mode === 'dark' 
              ? 'linear-gradient(135deg, #5b21b6 0%, #7c3aed 100%)' 
              : 'linear-gradient(135deg, #3730a3 0%, #6d28d9 100%)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backdropFilter: 'blur(12px)',
          border: mode === 'dark' ? '1px solid #334155' : '1px solid #e2e8f0',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '20px',
          boxShadow: mode === 'dark' 
            ? '0 4px 32px rgba(0, 0, 0, 0.3)' 
            : '0 4px 32px rgba(0, 0, 0, 0.06)',
          border: mode === 'dark' ? '1px solid #334155' : '1px solid #e2e8f0',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: '12px',
            '& fieldset': {
              borderColor: mode === 'dark' ? '#334155' : '#e2e8f0',
            },
          },
        },
      },
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          marginBottom: '4px',
          '&:hover': {
            backgroundColor: mode === 'dark' ? '#1e293b' : '#f1f5f9',
          },
        },
      },
    },
  },
});

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  const toggleTheme = () => {
    setDarkMode((prev: boolean) => {
      const newMode = !prev;
      localStorage.setItem('darkMode', JSON.stringify(newMode));
      return newMode;
    });
  };

  const theme = createAppTheme(darkMode ? 'dark' : 'light');

  const value = {
    darkMode,
    toggleTheme,
    theme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};
