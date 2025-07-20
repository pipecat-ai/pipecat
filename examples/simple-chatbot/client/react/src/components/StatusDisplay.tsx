import { usePipecatClientTransportState } from '@pipecat-ai/client-react';

export function StatusDisplay() {
  const transportState = usePipecatClientTransportState();

  return (
    <div className="status">
      Status: <span>{transportState}</span>
    </div>
  );
}
