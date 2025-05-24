import React, { useState, useEffect } from 'react';

interface ComputedMetrics {
  response_latency: {
    count: number;
    total: number;
    average: number;
    last: number;
  };
  interruption_latency: {
    count: number;
    total: number;
    average: number;
    last: number;
  };
}

interface ClientMetricsDisplayProps {
  computedMetrics: ComputedMetrics;
  showMetrics: boolean;
  sessionId: string | null;
  isActive: boolean;
  shouldTriggerAnalysis: string | null;
  onAnalysisTriggered: () => void;
}

interface ComparisonData {
  session_id: string;
  event_based_ms: number;
  audio_based_ms: number;
  variance_ms: number;
  variance_percent: number;
  status: 'accurate' | 'warning' | 'error';
}

// Add base URL constant
const API_BASE_URL = 'http://localhost:7860';

export const ClientMetricsDisplay: React.FC<ClientMetricsDisplayProps> = ({ 
  computedMetrics, 
  showMetrics, 
  sessionId, 
  isActive,
  shouldTriggerAnalysis,
  onAnalysisTriggered
}) => {
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debug logging
  useEffect(() => {
    console.log('🔍 ClientMetricsDisplay state:', {
      sessionId,
      isActive,
      isAnalyzing,
      analysisComplete,
      hasComparison: !!comparison,
      shouldTriggerAnalysis,
      error
    });
  }, [sessionId, isActive, isAnalyzing, analysisComplete, comparison, shouldTriggerAnalysis, error]);

  // Reset state when session changes
  useEffect(() => {
    console.log('🔄 Session changed, resetting state:', { sessionId });
    setComparison(null);
    setAnalysisComplete(false);
    setError(null);
    setIsAnalyzing(false);
  }, [sessionId]);

  // Clear comparison data when starting a new active session
  useEffect(() => {
    if (isActive) {
      console.log('🧹 New active session started, clearing comparison data');
      setComparison(null);
      setAnalysisComplete(false);
      setError(null);
      setIsAnalyzing(false);
    }
  }, [isActive]);

  // Additional clearing when computedMetrics get reset (when response/interruption metrics are cleared)
  useEffect(() => {
    if (computedMetrics.response_latency.count === 0 && computedMetrics.interruption_latency.count === 0) {
      console.log('🧹 Metrics cleared, clearing comparison data');
      setComparison(null);
      setAnalysisComplete(false);
      setError(null);
      setIsAnalyzing(false);
    }
  }, [computedMetrics]);

  // NEW: Trigger analysis when shouldTriggerAnalysis is set
  useEffect(() => {
    if (shouldTriggerAnalysis && shouldTriggerAnalysis === sessionId && !analysisComplete && !isAnalyzing) {
      console.log('🚀 Triggering audio analysis due to trigger signal for session:', shouldTriggerAnalysis);
      
      // Wait 5 seconds to ensure audio recording is fully saved
      const timer = setTimeout(() => {
        console.log('⏰ Timer fired, starting analysis for session:', shouldTriggerAnalysis);
        triggerAudioAnalysis();
        onAnalysisTriggered(); // Clear the trigger
      }, 5000);
      
      return () => {
        console.log('🧹 Cleaning up analysis timer');
        clearTimeout(timer);
      };
    }
  }, [shouldTriggerAnalysis, sessionId, analysisComplete, isAnalyzing, onAnalysisTriggered]);

  const triggerAudioAnalysis = async () => {
    if (!sessionId) {
      console.error('❌ No sessionId for analysis');
      return;
    }
    
    console.log(`🎵 Starting audio analysis for session: ${sessionId}`);
    setIsAnalyzing(true);
    setError(null);
    
    try {
      // Step 1: Trigger audio analysis - FIX: Add base URL
      console.log('📡 Sending POST request to analyze-audio...');
      const analyzeResponse = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/analyze-audio`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      console.log('📡 Analysis response:', analyzeResponse.status, analyzeResponse.statusText);
      
      if (!analyzeResponse.ok) {
        const errorText = await analyzeResponse.text();
        console.error('❌ Analysis failed:', errorText);
        throw new Error(`Analysis failed: ${analyzeResponse.statusText} - ${errorText}`);
      }
      
      const analyzeResult = await analyzeResponse.json();
      console.log('✅ Audio analysis completed:', analyzeResult);
      
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
      
      // FIX: Add base URL here too
      const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/latency-comparison`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch comparison: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('📈 Comparison data received:', data);
      
      if (data.status === 'success' && data.comparisons.length > 0) {
        setComparison(data.comparisons[0]);
        setAnalysisComplete(true);
      } else {
        setError(data.message || 'No comparison data available');
      }
      
    } catch (err) {
      console.error('❌ Error fetching comparison:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch comparison');
    }
  };

  const renderComparisonContent = () => {
    const metricRowStyle = {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '4px'
    };

    const labelStyle = {
      fontSize: '12px',
      color: '#6b7280'
    };

    const valueStyle = {
      fontSize: '14px',
      fontWeight: 'bold',
      color: '#1f2937'
    };

    if (isAnalyzing) {
      return (
        <div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Status:</span>
            <span style={valueStyle}>🔄 Analyzing...</span>
          </div>
        </div>
      );
    }

    if (error) {
      return (
        <div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Status:</span>
            <span style={{ ...valueStyle, color: '#ef4444' }}>❌ {error}</span>
          </div>
          <button 
            onClick={triggerAudioAnalysis}
            style={{
              marginTop: '8px',
              padding: '4px 8px',
              fontSize: '12px',
              backgroundColor: '#f3f4f6',
              border: '1px solid #d1d5db',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            🔄 Retry Analysis
          </button>
        </div>
      );
    }

    if (comparison) {
      return (
        <div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Variance:</span>
            <span style={valueStyle}>{comparison.variance_percent.toFixed(1)}%</span>
          </div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Event-based-Response-Latency:</span>
            <span style={valueStyle}>{comparison.event_based_ms.toFixed(0)}ms</span>
          </div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Audio-based-Response-Latency:</span>
            <span style={valueStyle}>{comparison.audio_based_ms.toFixed(0)}ms</span>
          </div>
        </div>
      );
    }

    if (!sessionId) {
      return (
        <div style={metricRowStyle}>
          <span style={labelStyle}>Status:</span>
          <span style={valueStyle}>Waiting...</span>
        </div>
      );
    }

    // Default state - waiting or session ended but no analysis yet
    return (
      <div>
        <div style={metricRowStyle}>
          <span style={labelStyle}>Status:</span>
          <span style={valueStyle}>Waiting...</span>
        </div>
        {!isActive && (
          <button 
            onClick={triggerAudioAnalysis}
            style={{
              marginTop: '8px',
              padding: '4px 8px',
              fontSize: '12px',
              backgroundColor: '#f3f4f6',
              border: '1px solid #d1d5db',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            🎵 Start Analysis
          </button>
        )}
      </div>
    );
  };

  if (!showMetrics) {
    return null;
  }

  const containerStyle: React.CSSProperties = {
    marginBottom: '20px',
    padding: '16px',
    backgroundColor: '#f8f9fa',
    border: '1px solid #dee2e6',
    borderRadius: '8px',
    fontFamily: 'monospace',
    fontSize: '14px'
  };

  const titleStyle: React.CSSProperties = {
    margin: '0 0 16px 0',
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#495057',
    textAlign: 'center'
  };

  const sectionsContainerStyle: React.CSSProperties = {
    display: 'flex',
    gap: '16px',
    justifyContent: 'space-between'
  };

  const sectionStyle: React.CSSProperties = {
    flex: 1,
    padding: '12px',
    backgroundColor: '#ffffff',
    border: '1px solid #e9ecef',
    borderRadius: '6px'
  };

  const sectionTitleStyle: React.CSSProperties = {
    margin: '0 0 12px 0',
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#212529'
  };

  const metricRowStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '6px'
  };

  const labelStyle: React.CSSProperties = {
    color: '#6c757d',
    fontSize: '13px'
  };

  const valueStyle: React.CSSProperties = {
    color: '#212529',
    fontWeight: 'bold',
    fontSize: '13px'
  };

  return (
    <div style={containerStyle}>
      <h3 style={titleStyle}>📊 Client Latency Metrics</h3>
      
      <div style={sectionsContainerStyle}>
        <div style={sectionStyle}>
          <h4 style={sectionTitleStyle}>⚡ Response Latency</h4>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Average:</span>
            <span style={valueStyle}>{computedMetrics.response_latency.average}ms</span>
          </div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Total:</span>
            <span style={valueStyle}>{computedMetrics.response_latency.total}ms</span>
          </div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Last:</span>
            <span style={valueStyle}>{computedMetrics.response_latency.last}ms</span>
          </div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Count:</span>
            <span style={valueStyle}>{computedMetrics.response_latency.count}</span>
          </div>
        </div>

        <div style={sectionStyle}>
          <h4 style={sectionTitleStyle}>🚫 Interruption Latency</h4>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Average:</span>
            <span style={valueStyle}>{computedMetrics.interruption_latency.average}ms</span>
          </div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Total:</span>
            <span style={valueStyle}>{computedMetrics.interruption_latency.total}ms</span>
          </div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Last:</span>
            <span style={valueStyle}>{computedMetrics.interruption_latency.last}ms</span>
          </div>
          <div style={metricRowStyle}>
            <span style={labelStyle}>Count:</span>
            <span style={valueStyle}>{computedMetrics.interruption_latency.count}</span>
          </div>
        </div>

        <div style={sectionStyle}>
          <h4 style={sectionTitleStyle}>🔍 Latency Validation</h4>
          {renderComparisonContent()}
        </div>
      </div>
    </div>
  );
}; 