Fixed the `multi_worker_handoff` Flows example repeating the assistant's reply
after handing control back to the router. `transfer_to_router` returned a next
node, so the just-deactivated reservation worker ran a second completion — and
wrote its node's tools — into the shared context alongside the router (its
directly queued frames bypass the bus activation gate), producing duplicate
replies and, on the way back, dropping `transfer_to_reservation` from the
router's tools. It now hands off with `NO_RESPONSE`, so the deactivated worker
neither responds nor touches the context.
