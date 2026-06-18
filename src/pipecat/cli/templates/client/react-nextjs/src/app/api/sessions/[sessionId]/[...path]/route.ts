import { NextResponse } from 'next/server';

async function handleRequest(
  request: Request,
  { params }: { params: Promise<{ sessionId: string; path: string[] }> }
) {
  const botBaseUrl =
    process.env.BOT_START_URL?.replace('/start', '') || 'http://localhost:7860';
  const { sessionId, path } = await params;
  const pathString = path.join('/');
  const targetUrl = `${botBaseUrl}/sessions/${sessionId}/${pathString}`;

  try {
    const body = await request.text();
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (process.env.BOT_START_PUBLIC_API_KEY) {
      headers.Authorization = `Bearer ${process.env.BOT_START_PUBLIC_API_KEY}`;
    }

    const response = await fetch(targetUrl, {
      method: request.method,
      headers,
      body,
    });

    if (!response.ok) {
      throw new Error(`Failed to proxy request: ${response.statusText}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error);
    }

    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to proxy session request: ${error}` },
      { status: 500 }
    );
  }
}

export async function POST(
  request: Request,
  context: { params: Promise<{ sessionId: string; path: string[] }> }
) {
  return handleRequest(request, context);
}

export async function PATCH(
  request: Request,
  context: { params: Promise<{ sessionId: string; path: string[] }> }
) {
  return handleRequest(request, context);
}
