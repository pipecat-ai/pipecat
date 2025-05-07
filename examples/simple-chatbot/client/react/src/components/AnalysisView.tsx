export function AnalysisView({
  audioUrl,
  onNewCall,
}: {
  audioUrl: string;
  onNewCall: () => void;
}) {
  return (
    <div className="analysis-view">
      <h2>Call Analysis</h2>
      <audio controls src={audioUrl} style={{ width: '100%' }} />
      {/* TODO: overlay latency markers & transcript here */}
      <button onClick={onNewCall} style={{ marginTop: '1rem' }}>
        Start New Call
      </button>
    </div>
  );
}
