import './globals.css';
import { RTVIProvider } from '@/providers/RTVIProvider';

export const metadata = {
  title: 'Pipecat React Client',
  description: 'Pipecat RTVI Client using Next.js',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <RTVIProvider>{children}</RTVIProvider>
      </body>
    </html>
  );
}
