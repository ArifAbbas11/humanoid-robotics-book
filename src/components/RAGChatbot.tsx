import React, { useState, useRef, useEffect } from 'react';
import styles from './RAGChatbot.module.css';

interface RetrievedChunk {
  content: string;
  similarity_score: number;
  source_url: string;
  chunk_id: string;
}

interface ResponseResult {
  answer: string;
  confidence_level: 'high' | 'medium' | 'low' | 'none';
  retrieved_chunks: RetrievedChunk[];
  processing_time: number;
  query_id: string;
}

const RAGChatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatPanelRef = useRef<HTMLDivElement>(null);

  // Get current page URL
  const currentPageUrl = typeof window !== 'undefined' ? window.location.href : '';

  // Function to get selected text
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().trim() !== '') {
        setSelectedText(selection.toString());
      } else {
        setSelectedText(null);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Toggle chat visibility
  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue;
    setInputValue('');

    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    setIsLoading(true);

    try {
      // Prepare the query based on context
      const requestBody = {
        question: userMessage,
        top_k: 5,  // Number of chunks to retrieve
        filters: null  // Optional metadata filters
      };

      // Call the backend API
      const response = await fetch('https://ayeshamasood110-rag-chatbot.hf.space/api/v1/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data: ResponseResult = await response.json();

      // Add assistant response to chat
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key press
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      {/* Floating button to open chat */}
      {!isOpen && (
        <button className={styles.chatButton} onClick={toggleChat}>
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        </button>
      )}

      {/* Chat panel */}
      {isOpen && (
        <div className={styles.chatContainer} ref={chatPanelRef}>
          <div className={styles.chatHeader}>
            <h3>Book Assistant</h3>
            <button className={styles.closeButton} onClick={toggleChat}>
              ×
            </button>
          </div>

          <div className={styles.chatMessages}>
            {messages.length === 0 ? (
              <div className={styles.welcomeMessage}>
                <p>Hello! I'm your book assistant. Ask me anything about the humanoid robotics content.</p>
                {selectedText && (
                  <p className={styles.selectedTextNotice}>
                    <strong>Selected text:</strong> "{selectedText.substring(0, 60)}{selectedText.length > 60 ? '...' : ''}"
                  </p>
                )}
              </div>
            ) : (
              messages.map((msg, index) => (
                <div key={index} className={`${styles.message} ${styles[msg.role]}`}>
                  <div className={styles.messageContent}>{msg.content}</div>
                </div>
              ))
            )}
            {isLoading && (
              <div className={`${styles.message} ${styles.assistant}`}>
                <div className={styles.messageContent}>Thinking...</div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className={styles.chatInputArea}>
            {selectedText && (
              <div className={styles.contextIndicator}>
                Using selected text: "{selectedText.substring(0, 50)}..."
              </div>
            )}
            <div className={styles.inputContainer}>
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about the book content..."
                className={styles.chatInput}
                rows={2}
              />
              <button
                onClick={handleSendMessage}
                disabled={isLoading || !inputValue.trim()}
                className={styles.sendButton}
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default RAGChatbot;