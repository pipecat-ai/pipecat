import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
    base: "./", //Use relative paths so it works at any mount path
    plugins: [react()],
    server: {
        proxy: {
            '/ws': {
                target: 'ws://0.0.0.0:8765', // Replace with your backend URL
                changeOrigin: true,
            },
        },
    },
});
