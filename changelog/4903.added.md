- Added support for outbound WhatsApp calling through the WhatsApp Cloud API.

  - New SmallWebRTCConnection.create_offer() generates an SDP offer for outbound connections.
  - New SmallWebRTCConnection.set_remote_description() accepts a remote SDP description.
  - WhatsAppClient.initiate_outbound_call() orchestrates the full outbound call flow.
  - WhatsAppClient._handle_connect_event() now handles outbound connects.
  - New POST /whatsapp/outbound endpoint for triggering outbound calls via HTTP.
