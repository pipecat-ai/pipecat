// [POST] /api

export async function POST(request: Request) {
  const params = await request.json();
    console.log("in POST, params is ", params)
    const url = process.env.BOT_START_URL || "http://localhost:7860"
  const req = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.PCC_API_KEY}`,
    },
    body: JSON.stringify(params),
  });

  const res = await req.json();

  if (req.status !== 200) {
    return Response.json(res, { status: req.status });
  }
  console.log({res});
  return Response.json(res);
}

export async function GET(request: Request) {
    return Response.json({message: "Hello World"});
}