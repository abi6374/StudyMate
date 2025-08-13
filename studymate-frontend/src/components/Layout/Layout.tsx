import React, { useState } from 'react';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Avatar,
  Menu,
  MenuItem,
  Switch,
  FormControlLabel,
  Tooltip,
  Badge,
  useMediaQuery,
  useTheme as useMuiTheme,
} from '@mui/material';
import {
  Menu as MenuIcon,
  CloudUpload,
  QuestionAnswer,
  Summarize,
  Quiz,
  Note,
  AutoStories,
  Settings,
  Logout,
  DarkMode,
  LightMode,
  Close,
  School,
} from '@mui/icons-material';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';

const drawerWidth = 280;

interface LayoutProps {
  children: React.ReactNode;
  activeTab?: number;
  onTabChange?: (tab: number) => void;
}

const Layout: React.FC<LayoutProps> = ({ children, activeTab = 0, onTabChange }) => {
  const { user, logout } = useAuth();
  const { darkMode, toggleTheme } = useTheme();
  const muiTheme = useMuiTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md'));
  
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    handleClose();
  };

  const navigationItems = [
    { 
      text: 'Upload & Index', 
      icon: <CloudUpload />, 
      index: 0, 
      description: 'Upload your PDFs' 
    },
    { 
      text: 'AI Chat', 
      icon: <QuestionAnswer />, 
      index: 1, 
      description: 'Ask questions',
      badge: 3 
    },
    { 
      text: 'Summaries', 
      icon: <Summarize />, 
      index: 2, 
      description: 'Document summaries' 
    },
    { 
      text: 'Flashcards', 
      icon: <AutoStories />, 
      index: 3, 
      description: 'Study cards' 
    },
    { 
      text: 'Quizzes', 
      icon: <Quiz />, 
      index: 4, 
      description: 'Test yourself' 
    },
    { 
      text: 'Notes', 
      icon: <Note />, 
      index: 5, 
      description: 'Your annotations' 
    },
  ];

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 3, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <School sx={{ fontSize: 32, color: 'primary.main' }} />
          <Typography variant="h5" fontWeight="bold" color="primary">
            StudyMate
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          Your AI-powered study companion
        </Typography>
      </Box>

      {/* Navigation */}
      <List sx={{ flex: 1, px: 2, py: 2 }}>
        {navigationItems.map((item) => (
          <ListItem key={item.index} disablePadding sx={{ mb: 1 }}>
            <ListItemButton
              selected={activeTab === item.index}
              onClick={() => {
                onTabChange?.(item.index);
                if (isMobile) setMobileOpen(false);
              }}
              sx={{
                borderRadius: 2,
                py: 1.5,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'white',
                  },
                },
                '&:hover': {
                  backgroundColor: darkMode ? 'grey.800' : 'grey.100',
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                {item.badge ? (
                  <Badge badgeContent={item.badge} color="error">
                    {item.icon}
                  </Badge>
                ) : (
                  item.icon
                )}
              </ListItemIcon>
              <ListItemText 
                primary={item.text}
                secondary={item.description}
                primaryTypographyProps={{
                  fontWeight: activeTab === item.index ? 600 : 500,
                  fontSize: '0.95rem',
                }}
                secondaryTypographyProps={{
                  fontSize: '0.8rem',
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      {/* Settings */}
      <Box sx={{ p: 2 }}>
        <FormControlLabel
          control={
            <Switch
              checked={darkMode}
              onChange={toggleTheme}
              color="primary"
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {darkMode ? <DarkMode fontSize="small" /> : <LightMode fontSize="small" />}
              <Typography variant="body2">
                {darkMode ? 'Dark Mode' : 'Light Mode'}
              </Typography>
            </Box>
          }
          sx={{ ml: 0 }}
        />
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { xs: '100%', md: `calc(100% - ${drawerWidth}px)` },
          ml: { xs: 0, md: `${drawerWidth}px` },
          backgroundColor: 'background.paper',
          color: 'text.primary',
          boxShadow: 'none',
          borderBottom: 1,
          borderColor: 'divider',
          zIndex: (theme) => theme.zIndex.drawer - 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Box sx={{ flexGrow: 1 }} />
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2" color="text.secondary">
              {user?.name}
            </Typography>
            <Tooltip title="Account settings">
              <IconButton
                size="large"
                onClick={handleMenu}
                color="inherit"
              >
                <Avatar sx={{ 
                  width: 36, 
                  height: 36, 
                  bgcolor: 'primary.main',
                  fontSize: '1rem',
                  fontWeight: 600,
                }}>
                  {user?.name?.charAt(0).toUpperCase()}
                </Avatar>
              </IconButton>
            </Tooltip>
            
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleClose}
              onClick={handleClose}
              PaperProps={{
                sx: {
                  mt: 1.5,
                  borderRadius: 2,
                  minWidth: 200,
                },
              }}
              transformOrigin={{ horizontal: 'right', vertical: 'top' }}
              anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
            >
              <MenuItem>
                <ListItemIcon>
                  <Settings fontSize="small" />
                </ListItemIcon>
                Settings
              </MenuItem>
              <MenuItem onClick={handleLogout}>
                <ListItemIcon>
                  <Logout fontSize="small" />
                </ListItemIcon>
                Logout
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Mobile Drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        ModalProps={{
          keepMounted: true,
        }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': { 
            boxSizing: 'border-box', 
            width: drawerWidth,
            backgroundColor: 'background.paper',
          },
        }}
      >
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <IconButton onClick={handleDrawerToggle}>
            <Close />
          </IconButton>
        </Box>
        {drawer}
      </Drawer>

      {/* Desktop Drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': { 
            boxSizing: 'border-box', 
            width: drawerWidth,
            backgroundColor: 'background.paper',
            border: 'none',
            borderRight: 1,
            borderColor: 'divider',
            zIndex: (theme) => theme.zIndex.drawer,
          },
        }}
        open
      >
        {drawer}
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { xs: '100%', md: `calc(100% - ${drawerWidth}px)` },
          ml: { xs: 0, md: `${drawerWidth}px` },
          height: '100vh',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}
        <Box sx={{ 
          flex: 1, 
          overflow: 'auto',
          backgroundColor: 'background.default',
          p: { xs: 1, sm: 2, md: 3 },
        }}>
          {children}
        </Box>
      </Box>
    </Box>
  );
};

export default Layout;
