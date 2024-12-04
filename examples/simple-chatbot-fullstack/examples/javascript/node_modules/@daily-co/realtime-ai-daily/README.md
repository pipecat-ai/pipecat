# Real-Time Voice Inference - Daily Transport

[![Docs](https://img.shields.io/badge/documentation-blue)](https://docs.rtvi.ai)
![NPM Version](https://img.shields.io/npm/v/@daily-co/realtime-ai-daily)

Daily transport package for use with `realtime-ai`.

## How to use

#### Install relevant packages

```bash
npm install realtime-ai @daily-co/realtime-ai-daily
```

#### Import and pass transport to your RTVI client
```typescript
import { RTVIClient } from "realtime-ai";
import { DailyTransport } from "@daily-co/realtime-ai-daily";

const rtviClient = new RTVIClient({
    transport: new DailyTransport(),
    // ... your RTVI config here
});

await rtviClient.connect();
```

###Documentation

Please refer to the RTVI documentation [here](https://docs.rtvi.ai).