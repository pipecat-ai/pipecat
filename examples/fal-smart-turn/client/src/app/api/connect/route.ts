import { NextResponse, NextRequest } from 'next/server';

export async function POST(request: NextRequest) {
  const { MY_CUSTOM_DATA } = await request.json();

  try {
    const response = await fetch(
      `https://api.pipecat.daily.co/v1/public/${process.env.AGENT_NAME}/start`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${process.env.PIPECAT_CLOUD_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          // Create Daily room
          createDailyRoom: true,
          // Optionally set Daily room properties
          dailyRoomProperties: { start_video_off: true },
          // Optionally pass custom data to the bot
          body: { MY_CUSTOM_DATA },
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`API responded with status: ${response.status}`);
    }

    const data = await response.json();

    // Transform the response to match what RTVI client expects
    return NextResponse.json({
      room_url: data.dailyRoom,
      token: data.dailyToken,
    });
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Failed to start agent' },
      { status: 500 }
    );
  }
}
