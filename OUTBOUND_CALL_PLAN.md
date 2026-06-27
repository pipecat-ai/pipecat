# Outbound WhatsApp Call — Implementation Plan

**Goal:** Add `WhatsAppClient.initiate_outbound_call(to_number)` so Pipecat can dial out to WhatsApp users.

## Background

- WhatsApp **Business Cloud API already supports outbound calls** via POST `/{phone_number_id}/calls`
- Pipecat already has all the plumbing: WebRTC `SmallWebRTCConnection`, SDP negotiation, WhatsApp API auth, webhook handling
- **Inbound flow:** User calls bot → Meta sends `connect` webhook → Pipecat creates WebRTC connection, generates SDP *answer*, sends `pre_accept` + `accept`
- **Outbound flow:** Bot calls user → Pipecat creates WebRTC connection, generates SDP *offer*, sends `offer` to Meta → User's phone rings → User answers → Meta sends `connect` webhook → Existing inbound handler takes over

---

## Files to Modify

### 1. `src/pipecat/transports/whatsapp/api.py` — Add `initiate_call_to_whatsapp()`

New method on `WhatsAppApi`:

```python
async def initiate_call_to_whatsapp(self, to_number: str, sdp: str, call_id: str) -> dict:
    """Initiate an outbound WhatsApp call.
    
    POSTs to Meta's /{phone_number_id}/calls with an SDP offer.
    Meta will ring the target WhatsApp user and upon answer,
    send a 'connect' webhook back to our /whatsapp endpoint.
    """
    async with self._session.post(
        self._whatsapp_url,
        headers={
            "Authorization": f"Bearer {self._whatsapp_token}",
            "Content-Type": "application/json",
        },
        json={
            "messaging_product": "whatsapp",
            "to": to_number,
            "action": "offer",
            "call_id": call_id,
            "session": {"sdp": sdp, "sdp_type": "offer"},
        },
    ) as response:
        return await response.json()
```

**Note:** `call_id` should be a UUID we generate server-side before sending the offer. Meta uses it to correlate the offer with the subsequent `connect` webhook when the user answers.

---

### 2. `src/pipecat/transports/whatsapp/client.py` — Add `initiate_outbound_call()`

New method on `WhatsAppClient`:

```python
async def initiate_outbound_call(
    self,
    to_number: str,
    connection_callback: Callable[..., Awaitable[None]] | None = None,
) -> str:
    """Initiate an outbound WhatsApp call to a target number.
    
    1. Creates a SmallWebRTC connection
    2. Generates an SDP offer
    3. Sends the offer to WhatsApp Cloud API
    4. Stores the connection for lifecycle management
    5. When user answers, Meta sends 'connect' webhook → existing handler
    
    Args:
        to_number: Target WhatsApp number (international format, no +)
        connection_callback: Same callback as handle_webhook_request()
    
    Returns:
        call_id: The generated call ID for tracking
    """
    import uuid
    
    call_id = str(uuid.uuid4())
    
    # 1. Create WebRTC connection
    pipecat_connection = SmallWebRTCConnection(self._ice_servers)
    await pipecat_connection.initialize()
    
    # 2. Generate SDP offer
    offer = pipecat_connection.get_offer()
    if offer is None:
        raise RuntimeError("Failed to generate SDP offer")
    
    sdp_offer = offer.get("sdp")
    if sdp_offer is None:
        raise RuntimeError("SDP offer missing 'sdp' field")
    
    sdp_offer = self._filter_sdp_for_whatsapp(sdp_offer)
    
    # 3. Send offer to WhatsApp API
    response = await self._whatsapp_api.initiate_call_to_whatsapp(
        to_number=to_number,
        sdp=sdp_offer,
        call_id=call_id,
    )
    
    if not response.get("success", False):
        await pipecat_connection.disconnect()
        raise RuntimeError(f"Failed to initiate call: {response}")
    
    # 4. Store the connection
    self._ongoing_calls_map[call_id] = pipecat_connection
    
    # Set up disconnect handler
    @pipecat_connection.event_handler("closed")
    async def handle_disconnected(webrtc_connection):
        self._ongoing_calls_map.pop(call_id, None)
    
    # 5. Register callback for when connect webhook arrives
    #    (stored so handle_webhook_request can find it)
    self._pending_outbound_calls[call_id] = {
        "callback": connection_callback,
        "to_number": to_number,
    }
    
    logger.debug(f"Outbound call initiated to {to_number}, call_id: {call_id}")
    return call_id
```

**New instance variable needed on `__init__`:**
```python
self._pending_outbound_calls: dict[str, dict] = {}
```

**Modification to `handle_webhook_request()`:**
When processing a `connect` event for a call_id that exists in `_pending_outbound_calls`, invoke the stored callback with the connection + call metadata. This is exactly the same flow as inbound calls — the webhook handler already creates the WebRTC connection and calls `connection_callback`.

**Wait — actually there's a subtlety.** For outbound, Meta sends the `connect` webhook *with a new SDP offer from the callee*. We need to:
1. Find the existing WebRTC connection from `_ongoing_calls_map` (or create a new one)
2. Set the remote SDP from the webhook
3. Generate an answer
4. Call `pre_accept` then `accept` (same as inbound)

**Simpler approach:** When the outbound `connect` webhook arrives, reuse the existing `_handle_connect_event()` — but skip creating a *new* connection, instead update the one we already made:

```python
# In handle_webhook_request, for connect events:
if call.id in self._pending_outbound_calls:
    # Outbound: update existing connection instead of creating new
    existing = self._ongoing_calls_map.get(call.id)
    if existing:
        await existing.set_remote_sdp(call.session.sdp, call.session.sdp_type)
        answer = existing.get_answer()
        # ... pre_accept + accept as normal
```

This way we reuse the existing connection + pipeline context.

---

### 3. `src/pipecat/runner/run.py` — Add `POST /whatsapp/outbound`

New route inside `_setup_whatsapp_routes()`:

```python
@app.post(
    "/whatsapp/outbound",
    summary="Initiate outbound WhatsApp call",
    description="Dials out to a WhatsApp number using the bot's pipeline",
)
async def initiate_outbound_call(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Initiate an outbound WhatsApp call.
    
    Body: {"to": "6281234567890", "bot_args": {}}
    """
    if whatsapp_client is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    body = await request.json()
    to_number = body.get("to")
    if not to_number:
        raise HTTPException(status_code=400, detail="Missing 'to' field")
    
    call_id = await whatsapp_client.initiate_outbound_call(
        to_number=to_number,
        connection_callback=connection_callback,  # Reuse same callback
    )
    
    return {"success": True, "call_id": call_id, "to": to_number}
```

---

### 4. `src/pipecat/transports/smallwebrtc/connection.py` — Check if `get_offer()` exists

Need to verify `SmallWebRTCConnection` has a method to generate SDP offers (not just answers). If it doesn't, we may need to add one that calls `createOffer()` on the underlying `RTCPeerConnection`.

---

### 5. Add WhatsApp outbound example

New file `examples/transports/transports-whatsapp-outbound.py` showing how to trigger an outbound call.

---

## Flow Summary

```
You tell Hermes: "Call Budi at 62812xxxxxxx"
  │
  ▼
Hermes calls POST /whatsapp/outbound {"to": "62812xxxxxxx"}
  │
  ▼
WhatsAppClient.initiate_outbound_call()
  ├── Creates SmallWebRTCConnection
  ├── Generates SDP offer
  ├── POSTS offer to Meta's WhatsApp Cloud API
  └── Stores connection in _ongoing_calls_map
  │
  ▼
Meta rings Budi's WhatsApp
  │
  ▼
Budi answers → Meta sends 'connect' webhook to our /whatsapp
  │
  ▼
WhatsAppClient.handle_webhook_request()
  ├── Detects this is a pending outbound call
  ├── Updates existing WebRTC connection with peer's SDP
  ├── Sends pre_accept → accept
  └── Invokes connection_callback → spawns bot pipeline
  │
  ▼
Bot talks to Budi naturally 🎉
```

## Deployment

```
python bot.py --whatsapp        # Starts FastAPI server
ngrok http 7860                 # Public URL for webhooks
curl -X POST localhost:7860/whatsapp/outbound \
  -H "Content-Type: application/json" \
  -d '{"to": "62812xxxxxxx"}'
```

## Risks & Considerations

1. **Call ID uniqueness**: Meta uses `call_id` to correlate offer ↔ answer. We need to generate a unique UUID server-side and keep it consistent.
2. **WebRTC connection lifecycle**: The outbound connection already exists in `_ongoing_calls_map` before the user answers. Need to make sure the `connect` webhook handler doesn't try to create a *new* connection.
3. **WhatsApp restrictions**: Outbound business calling is restricted in some regions. Indonesia/Bali should work.
4. **7-day window**: Once a user calls *you*, you can call them back within 7 days. For new conversations, Meta may have different rules.
5. **Error handling**: If the user doesn't answer, Meta sends a `terminate` webhook — already handled.
