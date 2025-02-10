import { defineConfig } from 'vite';

export default defineConfig({
    server: {
        proxy: {
            // Proxy /api requests to the backend server
            '/connect': {
                target: 'http://0.0.0.0:7860', // Replace with your backend URL
                changeOrigin: true,
            },
        },
    },
});
