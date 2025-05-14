import './globals.css';
import { RTVIProvider } from '@/providers/RTVIProvider';

export const metadata = {
  title: 'Pipecat React Client',
  description: 'Pipecat RTVI Client using Next.js',
  icons: {
    icon: [{ url: '/favicon.svg', type: 'image/svg+xml' }],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      </head>
      <body>
        <RTVIProvider>{children}</RTVIProvider>
      </body>
    </html>
  );
}
