import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
} from '@mui/material';
import Layout from '../Layout/Layout';
import UploadTab from './UploadTab';
import QATab from './QATab';
import SummaryTab from './SummaryTab';
import FlashcardsTab from './FlashcardsTab';
import QuizTab from './QuizTab';
import NotesTab from './NotesTab';

const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [conversationHistory, setConversationHistory] = useState<any[]>([]);

  const renderTabContent = () => {
    switch (activeTab) {
      case 0:
        return (
          <UploadTab 
            uploadedFiles={uploadedFiles} 
            setUploadedFiles={setUploadedFiles} 
          />
        );
      case 1:
        return (
          <QATab 
            uploadedFiles={uploadedFiles}
            conversationHistory={conversationHistory}
            setConversationHistory={setConversationHistory}
          />
        );
      case 2:
        return <SummaryTab uploadedFiles={uploadedFiles} />;
      case 3:
        return <FlashcardsTab uploadedFiles={uploadedFiles} />;
      case 4:
        return <QuizTab uploadedFiles={uploadedFiles} />;
      case 5:
        return <NotesTab />;
      default:
        return (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography variant="h5" color="text.secondary">
              Select a feature from the sidebar
            </Typography>
          </Box>
        );
    }
  };

  return (
    <Layout activeTab={activeTab} onTabChange={setActiveTab}>
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {renderTabContent()}
      </Box>
    </Layout>
  );
};

export default Dashboard;
