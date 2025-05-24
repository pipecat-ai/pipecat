import React, { useState, useEffect } from 'react';

interface ComparisonData {
  session_id: string;
  event_based_ms: number;
  audio_based_ms: number;
  variance_ms: number;
  variance_percent: number;
  status: 'accurate' | 'warning' | 'error';
}

interface LatencyComparisonResponse {
  session_id: string;
  status: string;
  message?: string;
  comparisons: ComparisonData[];
}

interface LatencyComparisonProps {
  sessionId: string | null;
  isActive: boolean;
}

export const LatencyComparison: React.FC<LatencyComparisonProps> = ({ 
  sessionId, 
  isActive 
}) => {
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'accurate': return '#22c55e'; // green
      case 'warning': return '#f59e0b';  // yellow
      case 'error': return '#ef4444';    // red
      default: return '#6b7280';         // gray
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'accurate': return '✅';
      case 'warning': return '⚠️';
      case 'error': return '❌';
      default: return '⏳';
    }
  };

  const triggerAudioAnalysis = async () => {
    if (!sessionId) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      console.log(`🎵 Triggering audio analysis for session: ${sessionId}`);
      
      // Step 1: Trigger audio analysis
      const analyzeResponse = await fetch(`/api/sessions/${sessionId}/analyze-audio`, {
        method: 'POST'
      });
      
      if (!analyzeResponse.ok) {
        throw new Error(`Analysis failed: ${analyzeResponse.statusText}`);
      }
      
      const analyzeResult = await analyzeResponse.json();
      console.log('📊 Audio analysis completed:', analyzeResult);
      
      // Step 2: Get comparison results
      await fetchComparison();
      
    } catch (err) {
      console.error('❌ Error during audio analysis:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fetchComparison = async () => {
    if (!sessionId) return;
    
    try {
      console.log(`📊 Fetching comparison for session: ${sessionId}`);
      
      const response = await fetch(`/api/sessions/${sessionId}/latency-comparison`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch comparison: ${response.statusText}`);
      }
      
      const data: LatencyComparisonResponse = await response.json();
      console.log('📈 Comparison data received:', data);
      
      if (data.status === 'success' && data.comparisons.length > 0) {
        setComparison(data.comparisons[0]); // Use first comparison for now
        setAnalysisComplete(true);
      } else {
        setError(data.message || 'No comparison data available');
      }
      
    } catch (err) {
      console.error('❌ Error fetching comparison:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch comparison');
    }
  };

  // Reset state when session changes
  useEffect(() => {
    setComparison(null);
    setAnalysisComplete(false);
    setError(null);
    setIsAnalyzing(false);
  }, [sessionId]);

  // Clear comparison data when starting a new active session
  useEffect(() => {
    if (isActive) {
      console.log('🧹 New active session started, clearing latency comparison data');
      setComparison(null);
      setAnalysisComplete(false);
      setError(null);
      setIsAnalyzing(false);
    }
  }, [isActive]);

  // Auto-trigger analysis when session becomes inactive (ends)
  useEffect(() => {
    if (!isActive && sessionId && !analysisComplete && !isAnalyzing) {
      // Wait a bit for recordings to be finalized
      const timer = setTimeout(() => {
        triggerAudioAnalysis();
      }, 2000);
      
      return () => clearTimeout(timer);
    }
  }, [isActive, sessionId, analysisComplete, isAnalyzing]);

  if (!sessionId) {
    return (
      <div className="latency-metric">
        <h3>🔍 Latency Validation</h3>
        <p className="metric-value">No session</p>
      </div>
    );
  }

  if (isAnalyzing) {
    return (
      <div className="latency-metric">
        <h3>🔍 Latency Validation</h3>
        <div className="metric-value analyzing">
          <span className="spinner">🔄</span>
          <span>Analyzing audio...</span>
        </div>
        <p className="metric-label">Comparing event vs audio data</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="latency-metric">
        <h3>🔍 Latency Validation</h3>
        <div className="metric-value error">
          <span>❌ {error}</span>
        </div>
        <button 
          className="retry-button"
          onClick={triggerAudioAnalysis}
          disabled={isAnalyzing}
        >
          🔄 Retry Analysis
        </button>
      </div>
    );
  }

  if (!comparison) {
    return (
      <div className="latency-metric">
        <h3>🔍 Latency Validation</h3>
        <div className="metric-value">
          <span>Waiting for session to end...</span>
        </div>
        <p className="metric-label">Will auto-analyze when complete</p>
      </div>
    );
  }

  return (
    <div className="latency-metric">
      <h3>🔍 Latency Validation</h3>
      <div className="metric-value" style={{ color: getStatusColor(comparison.status) }}>
        <div className="comparison-main">
          {getStatusIcon(comparison.status)} {comparison.status.toUpperCase()}
        </div>
        <div className="comparison-variance">
          ±{comparison.variance_ms.toFixed(0)}ms ({comparison.variance_percent.toFixed(1)}%)
        </div>
      </div>
      <div className="comparison-details">
        <div className="comparison-row">
          <span className="label">Event-based:</span>
          <span className="value">{comparison.event_based_ms.toFixed(0)}ms</span>
        </div>
        <div className="comparison-row">
          <span className="label">Audio-based:</span>
          <span className="value">{comparison.audio_based_ms.toFixed(0)}ms</span>
        </div>
      </div>
    </div>
  );
}; 