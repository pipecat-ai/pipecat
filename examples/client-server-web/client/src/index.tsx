import {
  ConsoleTemplate,
  FullScreenContainer,
  ThemeProvider,
} from '@pipecat-ai/voice-ui-kit';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

//@ts-ignore - fontsource-variable/geist is not typed
import '@fontsource-variable/geist';
//@ts-ignore - fontsource-variable/geist is not typed
import '@fontsource-variable/geist-mono';

createRoot(document.getElementById('root')!).render(
  // @ts-ignore
  <StrictMode>
    <ThemeProvider>
      <FullScreenContainer>
        <ConsoleTemplate
          transportType="smallwebrtc"
          connectParams={{
            connectionUrl: '/api/offer',
          }}
        />
      </FullScreenContainer>
    </ThemeProvider>
  </StrictMode>
);
