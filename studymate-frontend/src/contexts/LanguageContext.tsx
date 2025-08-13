import React, { createContext, useContext, useState, ReactNode } from 'react';

export interface LanguageContextType {
  language: string;
  setLanguage: (lang: string) => void;
  t: (key: string) => string;
}

interface Translations {
  [key: string]: {
    [key: string]: string;
  };
}

const translations: Translations = {
  en: {
    // Navigation
    'nav.upload': 'Upload Documents',
    'nav.chat': 'AI Chat',
    'nav.summary': 'Summaries',
    'nav.flashcards': 'Flashcards',
    'nav.quiz': 'Quiz',
    'nav.notes': 'Notes',
    'nav.logout': 'Logout',
    'nav.settings': 'Settings',
    
    // AI Chat
    'chat.title': 'AI-Powered Q&A Assistant',
    'chat.subtitle': 'Ask questions about your uploaded documents and get intelligent responses',
    'chat.placeholder': 'Ask a question about your documents...',
    'chat.send': 'Send',
    'chat.thinking': 'AI is analyzing your question...',
    'chat.error': 'Error getting response. Please try again.',
    'chat.noDocuments': 'No documents uploaded yet. Please upload a PDF first.',
    'chat.confidence': 'Confidence',
    'chat.processingTime': 'Processing time',
    'chat.language': 'Detected language',
    
    // Upload
    'upload.title': 'Upload PDF Documents',
    'upload.subtitle': 'Upload your PDF documents for AI-powered analysis',
    'upload.dragDrop': 'Drag and drop PDF files here, or click to select',
    'upload.processing': 'Processing documents...',
    'upload.success': 'Document uploaded successfully',
    'upload.error': 'Failed to upload document',
    
    // Settings
    'settings.title': 'Settings',
    'settings.language': 'Language',
    'settings.responseStyle': 'Response Style',
    'settings.theme': 'Theme',
    'settings.comprehensive': 'Comprehensive',
    'settings.concise': 'Concise',
    'settings.detailed': 'Detailed',
    
    // General
    'general.loading': 'Loading...',
    'general.error': 'An error occurred',
    'general.success': 'Success',
    'general.cancel': 'Cancel',
    'general.save': 'Save',
    'general.delete': 'Delete',
    'general.edit': 'Edit',
  },
  es: {
    // Navigation
    'nav.upload': 'Subir Documentos',
    'nav.chat': 'Chat IA',
    'nav.summary': 'Resúmenes',
    'nav.flashcards': 'Tarjetas',
    'nav.quiz': 'Cuestionario',
    'nav.notes': 'Notas',
    'nav.logout': 'Cerrar Sesión',
    'nav.settings': 'Configuración',
    
    // AI Chat
    'chat.title': 'Asistente de IA con Preguntas y Respuestas',
    'chat.subtitle': 'Haz preguntas sobre tus documentos y obtén respuestas inteligentes',
    'chat.placeholder': 'Haz una pregunta sobre tus documentos...',
    'chat.send': 'Enviar',
    'chat.thinking': 'La IA está analizando tu pregunta...',
    'chat.error': 'Error al obtener respuesta. Por favor, inténtalo de nuevo.',
    'chat.noDocuments': 'No hay documentos subidos. Por favor, sube un PDF primero.',
    'chat.confidence': 'Confianza',
    'chat.processingTime': 'Tiempo de procesamiento',
    'chat.language': 'Idioma detectado',
    
    // Upload
    'upload.title': 'Subir Documentos PDF',
    'upload.subtitle': 'Sube tus documentos PDF para análisis con IA',
    'upload.dragDrop': 'Arrastra y suelta archivos PDF aquí, o haz clic para seleccionar',
    'upload.processing': 'Procesando documentos...',
    'upload.success': 'Documento subido exitosamente',
    'upload.error': 'Error al subir documento',
    
    // Settings
    'settings.title': 'Configuración',
    'settings.language': 'Idioma',
    'settings.responseStyle': 'Estilo de Respuesta',
    'settings.theme': 'Tema',
    'settings.comprehensive': 'Completo',
    'settings.concise': 'Conciso',
    'settings.detailed': 'Detallado',
    
    // General
    'general.loading': 'Cargando...',
    'general.error': 'Ocurrió un error',
    'general.success': 'Éxito',
    'general.cancel': 'Cancelar',
    'general.save': 'Guardar',
    'general.delete': 'Eliminar',
    'general.edit': 'Editar',
  },
  fr: {
    // Navigation
    'nav.upload': 'Télécharger Documents',
    'nav.chat': 'Chat IA',
    'nav.summary': 'Résumés',
    'nav.flashcards': 'Cartes Mémoire',
    'nav.quiz': 'Quiz',
    'nav.notes': 'Notes',
    'nav.logout': 'Déconnexion',
    'nav.settings': 'Paramètres',
    
    // AI Chat
    'chat.title': 'Assistant IA Questions-Réponses',
    'chat.subtitle': 'Posez des questions sur vos documents et obtenez des réponses intelligentes',
    'chat.placeholder': 'Posez une question sur vos documents...',
    'chat.send': 'Envoyer',
    'chat.thinking': 'L\'IA analyse votre question...',
    'chat.error': 'Erreur lors de l\'obtention de la réponse. Veuillez réessayer.',
    'chat.noDocuments': 'Aucun document téléchargé. Veuillez d\'abord télécharger un PDF.',
    'chat.confidence': 'Confiance',
    'chat.processingTime': 'Temps de traitement',
    'chat.language': 'Langue détectée',
    
    // Upload
    'upload.title': 'Télécharger Documents PDF',
    'upload.subtitle': 'Téléchargez vos documents PDF pour une analyse par IA',
    'upload.dragDrop': 'Glissez-déposez les fichiers PDF ici, ou cliquez pour sélectionner',
    'upload.processing': 'Traitement des documents...',
    'upload.success': 'Document téléchargé avec succès',
    'upload.error': 'Échec du téléchargement du document',
    
    // Settings
    'settings.title': 'Paramètres',
    'settings.language': 'Langue',
    'settings.responseStyle': 'Style de Réponse',
    'settings.theme': 'Thème',
    'settings.comprehensive': 'Complet',
    'settings.concise': 'Concis',
    'settings.detailed': 'Détaillé',
    
    // General
    'general.loading': 'Chargement...',
    'general.error': 'Une erreur s\'est produite',
    'general.success': 'Succès',
    'general.cancel': 'Annuler',
    'general.save': 'Sauvegarder',
    'general.delete': 'Supprimer',
    'general.edit': 'Modifier',
  },
  hi: {
    // Navigation
    'nav.upload': 'दस्तावेज़ अपलोड करें',
    'nav.chat': 'AI चैट',
    'nav.summary': 'सारांश',
    'nav.flashcards': 'फ्लैशकार्ड',
    'nav.quiz': 'क्विज़',
    'nav.notes': 'नोट्स',
    'nav.logout': 'लॉग आउट',
    'nav.settings': 'सेटिंग्स',
    
    // AI Chat
    'chat.title': 'AI संचालित Q&A सहायक',
    'chat.subtitle': 'अपने अपलोड किए गए दस्तावेज़ों के बारे में प्रश्न पूछें और बुद्धिमान उत्तर प्राप्त करें',
    'chat.placeholder': 'अपने दस्तावेज़ों के बारे में एक प्रश्न पूछें...',
    'chat.send': 'भेजें',
    'chat.thinking': 'AI आपके प्रश्न का विश्लेषण कर रहा है...',
    'chat.error': 'उत्तर प्राप्त करने में त्रुटि। कृपया पुनः प्रयास करें।',
    'chat.noDocuments': 'अभी तक कोई दस्तावेज़ अपलोड नहीं किया गया। कृपया पहले एक PDF अपलोड करें।',
    'chat.confidence': 'विश्वास',
    'chat.processingTime': 'प्रसंस्करण समय',
    'chat.language': 'पहचानी गई भाषा',
    
    // Upload
    'upload.title': 'PDF दस्तावेज़ अपलोड करें',
    'upload.subtitle': 'AI-संचालित विश्लेषण के लिए अपने PDF दस्तावेज़ अपलोड करें',
    'upload.dragDrop': 'PDF फ़ाइलों को यहाँ खींचें और छोड़ें, या चुनने के लिए क्लिक करें',
    'upload.processing': 'दस्तावेज़ प्रसंस्करण...',
    'upload.success': 'दस्तावेज़ सफलतापूर्वक अपलोड किया गया',
    'upload.error': 'दस्तावेज़ अपलोड करने में विफल',
    
    // Settings
    'settings.title': 'सेटिंग्स',
    'settings.language': 'भाषा',
    'settings.responseStyle': 'उत्तर शैली',
    'settings.theme': 'थीम',
    'settings.comprehensive': 'व्यापक',
    'settings.concise': 'संक्षिप्त',
    'settings.detailed': 'विस्तृत',
    
    // General
    'general.loading': 'लोड हो रहा है...',
    'general.error': 'एक त्रुटि हुई',
    'general.success': 'सफलता',
    'general.cancel': 'रद्द करें',
    'general.save': 'सहेजें',
    'general.delete': 'हटाएं',
    'general.edit': 'संपादित करें',
  }
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<string>(() => {
    return localStorage.getItem('studymate-language') || 'en';
  });

  const handleSetLanguage = (lang: string) => {
    setLanguage(lang);
    localStorage.setItem('studymate-language', lang);
  };

  const t = (key: string): string => {
    return translations[language]?.[key] || translations['en'][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage: handleSetLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
