import React from 'react';

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
}

export const ClientMetricsDisplay: React.FC<ClientMetricsDisplayProps> = ({ computedMetrics, showMetrics }) => {
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
    flex: '1',
    padding: '12px',
    backgroundColor: '#ffffff',
    border: '1px solid #e9ecef',
    borderRadius: '6px'
  };

  const sectionTitleStyle: React.CSSProperties = {
    margin: '0 0 8px 0',
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#6c757d',
    textAlign: 'center'
  };

  const metricRowStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '4px'
  };

  const labelStyle: React.CSSProperties = {
    color: '#6c757d'
  };

  const valueStyle: React.CSSProperties = {
    fontWeight: 'bold',
    color: '#495057'
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
      </div>
    </div>
  );
}; 