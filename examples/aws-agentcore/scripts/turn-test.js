// Simple script for testing if we are correctly gathering the ice candidates from a specific
// turn server in the browser
(async () => {
  console.clear();
  console.log("Starting ICE candidate test…");

  const turnServer = {
    urls: "turn:turn.cloudflare.com:80?transport=tcp",
    username: "username",
    credential: "password"
  };

  const pc = new RTCPeerConnection({
    iceServers: [ turnServer ],
    iceTransportPolicy: "all"  // or "relay" if you want only TURN
  });

  let gotRelay = false;

  pc.onicecandidate = event => {
    if (!event.candidate) {
      console.log("ICE gathering finished.");
      if (gotRelay) {
        console.log("%cTURN relay candidate FOUND ✔️", "color: green; font-weight: bold;");
      } else {
        console.log("%cNo TURN relay candidates detected ❌", "color: red; font-weight: bold;");
      }
      return;
    }

    const cand = event.candidate.candidate;
    console.log("ICE Candidate:", cand);

    if (cand.includes("typ relay")) {
      console.log("%cTURN relay candidate detected!", "color: green;");
      gotRelay = true;
    }
  };

  // Create empty data channel (required to trigger ICE)
  pc.createDataChannel("test");

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  console.log("Gathering ICE candidates…");
})();