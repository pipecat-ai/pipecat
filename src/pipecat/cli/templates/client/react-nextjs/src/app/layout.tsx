import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'Pipecat Bot',
  icons: { icon: '/pipecat.svg' },
};

// Follows the system color scheme before first paint; the stylesheet keys its
// dark palette off the `dark` class.
const colorSchemeScript = `
  const scheme = window.matchMedia('(prefers-color-scheme: dark)');
  const applyScheme = () =>
    document.documentElement.classList.toggle('dark', scheme.matches);
  applyScheme();
  scheme.addEventListener('change', applyScheme);
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="color-scheme" content="light dark" />
        <script dangerouslySetInnerHTML={{ __html: colorSchemeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
