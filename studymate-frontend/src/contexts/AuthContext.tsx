import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

interface User {
  id: string;
  email: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Configure axios defaults
axios.defaults.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in on app start
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (token && userData) {
      setUser(JSON.parse(userData));
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    }
    setLoading(false);
  }, []);

  const login = async (email: string, password: string) => {
    try {
      // For demo purposes, simulate API call
      const response = await simulateAuthAPI({ email, password }, 'login');
      
      const { user: userData, token } = response.data;
      setUser(userData);
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(userData));
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } catch (error) {
      throw new Error('Invalid credentials');
    }
  };

  const register = async (name: string, email: string, password: string) => {
    try {
      // For demo purposes, simulate API call
      const response = await simulateAuthAPI({ name, email, password }, 'register');
      
      const { user: userData, token } = response.data;
      setUser(userData);
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(userData));
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } catch (error) {
      throw new Error('Registration failed');
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    delete axios.defaults.headers.common['Authorization'];
  };

  const value = {
    user,
    login,
    register,
    logout,
    loading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Simulate API calls for demo (replace with actual API calls)
const simulateAuthAPI = (credentials: any, type: 'login' | 'register'): Promise<any> => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (type === 'login') {
        if (credentials.email === 'demo@studymate.com' && credentials.password === 'demo123') {
          resolve({
            data: {
              user: { id: '1', email: 'demo@studymate.com', name: 'Demo User' },
              token: 'demo-token-123',
            },
          });
        } else {
          reject(new Error('Invalid credentials'));
        }
      } else if (type === 'register') {
        // Simple validation
        if (credentials.email && credentials.password && credentials.name) {
          resolve({
            data: {
              user: { id: '2', email: credentials.email, name: credentials.name },
              token: 'demo-token-456',
            },
          });
        } else {
          reject(new Error('Registration failed'));
        }
      }
    }, 1000); // Simulate network delay
  });
};
