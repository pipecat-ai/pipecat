'use client';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { TRANSPORT_LABELS, type TransportType } from '@/config';

interface TransportSelectProps {
  transportType: TransportType;
  onTransportChange: (type: TransportType) => void;
  availableTransports: TransportType[];
}

/** Compact transport picker for the console header. */
export const TransportSelect = ({
  transportType,
  onTransportChange,
  availableTransports,
}: TransportSelectProps) => {
  const items = availableTransports.map((transport) => ({
    value: transport,
    label: TRANSPORT_LABELS[transport],
  }));

  return (
    <Select
      items={items}
      value={transportType}
      onValueChange={(value) => onTransportChange(value as TransportType)}>
      <SelectTrigger size="sm" aria-label="Transport" className="w-40 min-w-24">
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="end">
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};
