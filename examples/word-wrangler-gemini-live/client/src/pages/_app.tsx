import { ConfigurationProvider } from '@/contexts/Configuration';
import { PipecatProvider } from '@/providers/PipecatProvider';
import { PipecatClientAudio } from '@pipecat-ai/client-react';
import type { AppProps } from 'next/app';
import { Nunito } from 'next/font/google';
import Head from 'next/head';
import '../styles/globals.css';

const nunito = Nunito({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-sans',
});

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>Daily | Word Wrangler</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <main className={`${nunito.variable}`}>
        <ConfigurationProvider>
          <PipecatProvider>
            <PipecatClientAudio />
            <Component {...pageProps} />
          </PipecatProvider>
        </ConfigurationProvider>
      </main>
    </>
  );
}
