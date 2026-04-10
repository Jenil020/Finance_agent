// src/utils/helpers.js
import { v4 as uuidv4 } from 'uuid';

/**
 * Generate a unique session ID
 */
export const generateSessionId = () => {
  return uuidv4();
};

/**
 * Format portfolio for API request
 * @param {Array} portfolio - Array of portfolio items
 * @returns {Array} Formatted portfolio
 */
export const formatPortfolio = (portfolio) => {
  return portfolio.map(item => ({
    ticker: item.ticker?.toUpperCase() || '',
    quantity: parseFloat(item.quantity) || 0,
    avg_cost: parseFloat(item.avg_cost) || 0,
  })).filter(item => item.ticker); // Remove empty entries
};

/**
 * Parse streaming response
 * @param {string} chunk - Response chunk
 * @returns {string} Cleaned text
 */
export const parseStreamChunk = (chunk) => {
  // Handle different streaming formats
  if (chunk.startsWith('data: ')) {
    return chunk.slice(6);
  }
  return chunk;
};

/**
 * Format currency value
 * @param {number} value - Value to format
 * @returns {string} Formatted currency
 */
export const formatCurrency = (value) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(value);
};

/**
 * Calculate portfolio value
 * @param {Array} portfolio - Array of portfolio items
 * @returns {number} Total portfolio value
 */
export const calculatePortfolioValue = (portfolio) => {
  return portfolio.reduce((total, item) => {
    return total + (item.quantity * item.avg_cost);
  }, 0);
};

/**
 * Validate portfolio item
 * @param {Object} item - Portfolio item
 * @returns {boolean} Is valid
 */
export const isValidPortfolioItem = (item) => {
  return (
    item.ticker &&
    item.ticker.trim().length > 0 &&
    !isNaN(item.quantity) &&
    item.quantity > 0 &&
    !isNaN(item.avg_cost) &&
    item.avg_cost > 0
  );
};

/**
 * Extract sources from agent trace
 * @param {Array} agentTrace - Agent trace from API response
 * @returns {Array} Extracted sources
 */
export const extractSources = (agentTrace) => {
  if (!Array.isArray(agentTrace)) return [];
  
  return agentTrace.reduce((sources, trace) => {
    if (trace.source) {
      sources.push(trace.source);
    }
    return sources;
  }, []);
};

/**
 * Format timestamp
 * @param {string|Date} date - Date to format
 * @returns {string} Formatted date
 */
export const formatDate = (date) => {
  return new Date(date).toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

/**
 * Retry logic for failed requests
 * @param {Function} fn - Async function to retry
 * @param {number} maxRetries - Max retry attempts
 * @returns {Promise} Result from function
 */
export const withRetry = async (fn, maxRetries = 3) => {
  let lastError;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      // Wait before retry (exponential backoff)
      await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
    }
  }
  
  throw lastError;
};