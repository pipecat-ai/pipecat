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
    <label className="flex items-center gap-2">
      <span className="text-muted-foreground text-xs">Transport</span>
      <Select
        items={items}
        value={transportType}
        onValueChange={(value) => onTransportChange(value as TransportType)}>
        <SelectTrigger size="sm">
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
    </label>
  );
};
