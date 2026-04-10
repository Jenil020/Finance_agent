// src/services/api.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_VERSION = import.meta.env.VITE_API_VERSION || 'v1';

const API_ENDPOINT = `${API_BASE_URL}/api/${API_VERSION}`;

// Create axios instance
const apiClient = axios.create({
  baseURL: API_ENDPOINT,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Health check - verify backend is running
 */
export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/health');
    return { status: response.status, data: response.data };
  } catch (error) {
    console.error('Health check failed:', error.message);
    throw error;
  }
};

/**
 * Non-streaming chat - get full response with sources
 * @param {Object} payload - { query, session_id, portfolio }
 * @returns {Promise<Object>} { answer, sources, agent_trace, session_id }
 */
export const chatNonStreaming = async (payload) => {
  try {
    const response = await apiClient.post('/chat/', payload);
    return response.data;
  } catch (error) {
    console.error('Chat error:', error.message);
    throw error;
  }
};

/**
 * Streaming chat - real-time response
 * @param {Object} payload - { query, session_id, portfolio }
 * @returns {Promise<ReadableStream>}
 */
export const chatStreaming = async (payload) => {
  try {
    const response = await fetch(`${API_ENDPOINT}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Streaming failed with status ${response.status}`);
    }

    return response.body.getReader();
  } catch (error) {
    console.error('Streaming error:', error.message);
    throw error;
  }
};

/**
 * Ingest files - upload documents to vector database
 * @param {Object} payload - { file_paths, metadata }
 * @returns {Promise<Object>}
 */
export const ingestFiles = async (payload) => {
  try {
    const response = await apiClient.post('/ingest', payload);
    return response.data;
  } catch (error) {
    console.error('Ingest error:', error.message);
    throw error;
  }
};

/**
 * Clear collection - delete from vector database
 * @param {string} collection - Collection name/ID
 * @returns {Promise<Object>}
 */
export const clearCollection = async (collection) => {
  try {
    const response = await apiClient.delete(`/ingest/${collection}`);
    return response.data;
  } catch (error) {
    console.error('Clear collection error:', error.message);
    throw error;
  }
};

export default apiClient;