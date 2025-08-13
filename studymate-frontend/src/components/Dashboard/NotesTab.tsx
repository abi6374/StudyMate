import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Card,
  CardContent,
  CardActions,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Fab,
} from '@mui/material';
import {
  Note,
  Add,
  Edit,
  Delete,
  Search,
  BookmarkBorder,
  Bookmark,
} from '@mui/icons-material';
import { toast } from 'react-toastify';

interface NoteItem {
  id: string;
  title: string;
  content: string;
  tags: string[];
  timestamp: Date;
  isBookmarked: boolean;
}

const NotesTab: React.FC = () => {
  const [notes, setNotes] = useState<NoteItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingNote, setEditingNote] = useState<NoteItem | null>(null);
  const [newNote, setNewNote] = useState({
    title: '',
    content: '',
    tags: '',
  });

  const filteredNotes = notes.filter(note =>
    note.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    note.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
    note.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const handleSaveNote = () => {
    if (!newNote.title.trim() || !newNote.content.trim()) {
      toast.error('Please fill in both title and content');
      return;
    }

    const noteToSave: NoteItem = {
      id: editingNote?.id || Date.now().toString(),
      title: newNote.title.trim(),
      content: newNote.content.trim(),
      tags: newNote.tags.split(',').map(tag => tag.trim()).filter(tag => tag),
      timestamp: editingNote?.timestamp || new Date(),
      isBookmarked: editingNote?.isBookmarked || false,
    };

    if (editingNote) {
      setNotes(prev => prev.map(note => 
        note.id === editingNote.id ? noteToSave : note
      ));
      toast.success('Note updated successfully!');
    } else {
      setNotes(prev => [noteToSave, ...prev]);
      toast.success('Note created successfully!');
    }

    handleCloseDialog();
  };

  const handleEditNote = (note: NoteItem) => {
    setEditingNote(note);
    setNewNote({
      title: note.title,
      content: note.content,
      tags: note.tags.join(', '),
    });
    setDialogOpen(true);
  };

  const handleDeleteNote = (noteId: string) => {
    setNotes(prev => prev.filter(note => note.id !== noteId));
    toast.info('Note deleted');
  };

  const handleBookmarkToggle = (noteId: string) => {
    setNotes(prev => prev.map(note =>
      note.id === noteId 
        ? { ...note, isBookmarked: !note.isBookmarked }
        : note
    ));
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setEditingNote(null);
    setNewNote({ title: '', content: '', tags: '' });
  };

  const handleNewNote = () => {
    setEditingNote(null);
    setNewNote({ title: '', content: '', tags: '' });
    setDialogOpen(true);
  };

  return (
    <Box sx={{ p: 3, position: 'relative', minHeight: '100%' }}>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        Notes & Annotations
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Save and organize your study notes alongside your documents
      </Typography>

      {/* Search Bar */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <TextField
          fullWidth
          placeholder="Search notes..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
          }}
        />
      </Paper>

      {/* Notes List */}
      {filteredNotes.length > 0 ? (
        <Box sx={{ display: 'grid', gap: 2 }}>
          {filteredNotes.map((note) => (
            <Card key={note.id} sx={{ position: 'relative' }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, flexGrow: 1 }}>
                    {note.title}
                  </Typography>
                  <IconButton
                    size="small"
                    onClick={() => handleBookmarkToggle(note.id)}
                    sx={{ ml: 1 }}
                  >
                    {note.isBookmarked ? (
                      <Bookmark color="primary" />
                    ) : (
                      <BookmarkBorder />
                    )}
                  </IconButton>
                </Box>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2, lineHeight: 1.6 }}>
                  {note.content.length > 200 
                    ? `${note.content.substring(0, 200)}...` 
                    : note.content
                  }
                </Typography>

                {note.tags.length > 0 && (
                  <Box sx={{ mb: 2, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {note.tags.map((tag, index) => (
                      <Chip
                        key={index}
                        label={tag}
                        size="small"
                        variant="outlined"
                        sx={{ fontSize: '0.75rem' }}
                      />
                    ))}
                  </Box>
                )}

                <Typography variant="caption" color="text.secondary">
                  {note.timestamp.toLocaleString()}
                </Typography>
              </CardContent>

              <CardActions sx={{ justifyContent: 'flex-end' }}>
                <Button
                  size="small"
                  startIcon={<Edit />}
                  onClick={() => handleEditNote(note)}
                >
                  Edit
                </Button>
                <Button
                  size="small"
                  startIcon={<Delete />}
                  color="error"
                  onClick={() => handleDeleteNote(note.id)}
                >
                  Delete
                </Button>
              </CardActions>
            </Card>
          ))}
        </Box>
      ) : notes.length > 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.50' }}>
          <Search sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="body1" color="text.secondary">
            No notes found matching "{searchTerm}"
          </Typography>
        </Paper>
      ) : (
        <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.50' }}>
          <Note sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No Notes Yet
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
            Start creating notes to organize your thoughts and insights
          </Typography>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={handleNewNote}
          >
            Create First Note
          </Button>
        </Paper>
      )}

      {/* Floating Action Button */}
      <Fab
        color="primary"
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
        }}
        onClick={handleNewNote}
      >
        <Add />
      </Fab>

      {/* Note Dialog */}
      <Dialog 
        open={dialogOpen} 
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingNote ? 'Edit Note' : 'Create New Note'}
        </DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Title"
            value={newNote.title}
            onChange={(e) => setNewNote(prev => ({ ...prev, title: e.target.value }))}
            sx={{ mb: 2, mt: 1 }}
          />
          
          <TextField
            fullWidth
            label="Content"
            multiline
            rows={8}
            value={newNote.content}
            onChange={(e) => setNewNote(prev => ({ ...prev, content: e.target.value }))}
            sx={{ mb: 2 }}
          />
          
          <TextField
            fullWidth
            label="Tags (comma-separated)"
            value={newNote.tags}
            onChange={(e) => setNewNote(prev => ({ ...prev, tags: e.target.value }))}
            placeholder="study, important, chapter1"
            helperText="Add tags to organize your notes"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>
            Cancel
          </Button>
          <Button 
            onClick={handleSaveNote}
            variant="contained"
          >
            {editingNote ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default NotesTab;
