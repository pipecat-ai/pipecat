import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  SelectGuide,
} from '@pipecat-ai/voice-ui-kit';

import { TRANSPORT_LABELS, type TransportType } from '../config';

interface TransportSelectProps {
  transportType: TransportType;
  onTransportChange: (type: TransportType) => void;
  availableTransports: TransportType[];
}

export const TransportSelect = ({
  transportType,
  onTransportChange,
  availableTransports,
}: TransportSelectProps) => {
  return (
    <Select value={transportType} onValueChange={onTransportChange}>
      <SelectTrigger size="lg">
        <SelectGuide>Transport</SelectGuide>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {availableTransports.map((transport) => (
          <SelectItem key={transport} value={transport}>
            {TRANSPORT_LABELS[transport]}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};
