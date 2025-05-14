import { Head, Html, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <meta
          name="description"
          content="Describe words without saying them and an AI will guess them!"
        />
        <link rel="icon" href="/favicon.ico" />
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
        <meta charSet="UTF-8" />

        {/* Open Graph / Social Media Meta Tags */}
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://word-wrangler.vercel.app/" />
        <meta
          property="og:title"
          content="Word Wrangler - AI Word Guessing Game"
        />
        <meta
          property="og:description"
          content="Describe words without saying them and an AI will guess them!"
        />
        <meta property="og:image" content="/og-image.png" />

        {/* Twitter Card Meta Tags */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:url" content="https://word-wrangler.vercel.app/" />
        <meta
          name="twitter:title"
          content="Word Wrangler - AI Word Guessing Game"
        />
        <meta
          name="twitter:description"
          content="Describe words without saying them and an AI will guess them!"
        />
        <meta name="twitter:image" content="/og-image.png" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
