import { PipecatLogo } from '@/components/pipecat/console/logo';

interface HeaderBrandProps {
  name: string;
}

/**
 * Logo and project name centered in the console header. The console renders
 * its `logo` slot at the left edge, so this positions itself against the
 * console root instead; the header's vertical padding and control height
 * give the offsets.
 */
export const HeaderBrand = ({ name }: HeaderBrandProps) => {
  return (
    <span className="pointer-events-none absolute top-2 left-1/2 flex h-7 -translate-x-1/2 items-center gap-2">
      <PipecatLogo height={20} />
      <strong className="text-sm whitespace-nowrap">{name}</strong>
    </span>
  );
};
