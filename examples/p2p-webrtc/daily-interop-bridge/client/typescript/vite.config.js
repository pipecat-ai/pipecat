import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
    plugins: [react()],
    server: {
        allowedHosts: true, // Allows external connections like ngrok
        proxy: {
            // Proxy /api requests to the backend server
            '/api': {
                target: 'http://0.0.0.0:7860', // Replace with your backend URL
                changeOrigin: true,
            },
        },
    },
});
