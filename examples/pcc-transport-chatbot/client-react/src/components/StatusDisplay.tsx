import { useRTVIClientTransportState } from '@pipecat-ai/client-react';

export function StatusDisplay() {
  const transportState = useRTVIClientTransportState();

  return (
    <div className="status">
      Status: <span>{transportState}</span>
    </div>
  );
}
