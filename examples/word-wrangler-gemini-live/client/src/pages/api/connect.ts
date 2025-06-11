import type { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { personality } = req.body;

    // Validate required parameters
    if (!personality) {
      return res
        .status(400)
        .json({ error: 'Missing required configuration parameters' });
    }

    const response = await fetch(
      `https://api.pipecat.daily.co/v1/public/${process.env.AGENT_NAME}/start`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${process.env.PIPECAT_CLOUD_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          createDailyRoom: true,
          body: {
            personality,
          },
        }),
      }
    );

    const data = await response.json();

    console.log('Response from API:', JSON.stringify(data, null, 2));

    // Transform the response to match what Pipecat client expects
    return res.status(200).json({
      room_url: data.dailyRoom,
      token: data.dailyToken,
    });
  } catch (error) {
    console.error('Error starting agent:', error);
    return res.status(500).json({ error: 'Failed to start agent' });
  }
}
