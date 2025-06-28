import { useState, useEffect } from 'react';

interface Item {
  id: string;
  type: 'image' | 'video';
  title: string;
  description: string;
  url: string;
}

// Determine backend base URL
const runtimeDefault = typeof window !== 'undefined'
  ? `${window.location.protocol}//${window.location.hostname}:27880`
  : '';
const BACKEND_BASE = (process.env.NEXT_PUBLIC_BACKEND_URL || runtimeDefault).replace(/\/$/, '');

interface Props {
  clientId: string;
  open: boolean;
  onClose: () => void;
}

// ---------------- PROJECTS MODAL (Gallery) ---------------- //
// Lightweight modal overlay that lists images/videos from gallery_metadata.json
// Located in backend assets. Selecting an item posts the selection to the backend
// REST endpoint /select-project so the LLM can access the context.
export default function ProjectsModal({ clientId, open, onClose }: Props) {
  const [items, setItems] = useState<Item[]>([]);

  useEffect(() => {
    if (!open) return;
    // Fetch metadata when modal opens. Assumes same-origin backend serves the file.
    fetch(`${BACKEND_BASE}/assets/gallery_metadata.json`)
      .then((r) => r.json())
      .then((data: Item[]) => setItems(data))
      .catch((err) => console.error('[ProjectsModal] metadata fetch failed', err));
  }, [open]);

  async function select(id: string) {
    try {
      await fetch('/select-project', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: clientId, project_id: id }),
      });
    } catch (err) {
      console.error('[ProjectsModal] selection POST failed', err);
    }
    onClose();
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="bg-white text-black w-full max-w-3xl max-h-[80vh] overflow-y-auto rounded-lg p-6 shadow-lg">
        <h2 className="text-xl font-semibold mb-4">Projects</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {items.map((item) => (
            <button
              key={item.id}
              className="flex flex-col items-center gap-2 hover:opacity-80"
              onClick={() => select(item.id)}
            >
              {item.type === 'image' ? (
                <img
                  src={item.url}
                  alt={item.title}
                  className="h-32 w-full object-cover rounded"
                />
              ) : (
                <video
                  src={`${BACKEND_BASE}${item.url.startsWith('/') ? item.url : '/' + item.url}`}
                  className="h-32 w-full object-cover rounded"
                  muted
                  playsInline
                />
              )}
              <span className="text-sm text-center px-1">{item.title}</span>
            </button>
          ))}
        </div>
        <button
          className="mt-6 px-4 py-2 bg-gray-200 rounded w-full"
          onClick={onClose}
        >
          Close
        </button>
      </div>
    </div>
  );
}
