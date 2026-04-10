// src/hooks/useChat.js
import { useState, useCallback, useRef, useEffect } from 'react';
import { chatStreaming, chatNonStreaming } from '../services/api';
import { generateSessionId, formatPortfolio, parseStreamChunk } from '../utils/helpers';

/**
 * Custom hook for chat management
 * Handles streaming, non-streaming, session, and message history
 */
export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [streamingText, setStreamingText] = useState('');
  const abortControllerRef = useRef(null);

  // Initialize session ID on mount
  useEffect(() => {
    const storedSessionId = localStorage.getItem('chat_session_id');
    if (storedSessionId) {
      setSessionId(storedSessionId);
    } else {
      const newSessionId = generateSessionId();
      setSessionId(newSessionId);
      localStorage.setItem('chat_session_id', newSessionId);
    }
  }, []);

  /**
   * Send a streaming message
   */
  const sendStreamingMessage = useCallback(async (query, portfolio = []) => {
    setError(null);
    setIsLoading(true);
    setStreamingText('');

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: query,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      abortControllerRef.current = new AbortController();

      const payload = {
        query,
        session_id: sessionId,
        portfolio: formatPortfolio(portfolio),
      };

      const reader = await chatStreaming(payload);
      let fullResponse = '';

      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: streamDone } = await reader.read();
        done = streamDone;

        if (value) {
          const chunk = decoder.decode(value);
          const text = parseStreamChunk(chunk);
          fullResponse += text;
          setStreamingText(fullResponse);
        }
      }

      // Add assistant message
      const assistantMessage = {
        id: Date.now(),
        role: 'assistant',
        content: fullResponse,
        timestamp: new Date(),
        streaming: true,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setStreamingText('');
    } catch (err) {
      if (err.name !== 'AbortError') {
        const errorMsg = {
          id: Date.now(),
          role: 'assistant',
          content: `Error: ${err.message}`,
          timestamp: new Date(),
          isError: true,
        };
        setMessages(prev => [...prev, errorMsg]);
        setError(err.message);
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [sessionId]);

  /**
   * Send a non-streaming message (get full response with sources)
   */
  const sendMessage = useCallback(async (query, portfolio = []) => {
    setError(null);
    setIsLoading(true);

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: query,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      const payload = {
        query,
        session_id: sessionId,
        portfolio: formatPortfolio(portfolio),
      };

      const response = await chatNonStreaming(payload);

      const assistantMessage = {
        id: Date.now(),
        role: 'assistant',
        content: response.answer || '',
        timestamp: new Date(),
        sources: response.sources || [],
        agentTrace: response.agent_trace || [],
      };

      setMessages(prev => [...prev, assistantMessage]);
      return response;
    } catch (err) {
      const errorMsg = {
        id: Date.now(),
        role: 'assistant',
        content: `Error: ${err.message}`,
        timestamp: new Date(),
        isError: true,
      };
      setMessages(prev => [...prev, errorMsg]);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  /**
   * Cancel streaming request
   */
  const cancelStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
    }
  }, []);

  /**
   * Clear chat history
   */
  const clearHistory = useCallback(() => {
    setMessages([]);
    setStreamingText('');
    setError(null);
  }, []);

  /**
   * Start new session
   */
  const newSession = useCallback(() => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    localStorage.setItem('chat_session_id', newSessionId);
    clearHistory();
  }, [clearHistory]);

  return {
    // State
    messages,
    sessionId,
    isLoading,
    error,
    streamingText,

    // Methods
    sendMessage,
    sendStreamingMessage,
    cancelStream,
    clearHistory,
    newSession,
  };
};

export default useChat;