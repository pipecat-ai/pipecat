"use client";

import type {
  APIRequest,
  PipecatClientOptions,
  TransportConnectionParams,
} from "@pipecat-ai/client-js";
import { PipecatClient } from "@pipecat-ai/client-js";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  createTransport,
  type TransportFactory,
  type TransportOptions,
  type TransportType,
} from "@/lib/transports";

/** Client states from which a new connection attempt may start. */
const CONNECTABLE_STATES: string[] = ["initialized", "disconnected", "error"];

export interface UsePipecatAppOptions {
  /** Defaults to smallwebrtc; changing this rebuilds the client. */
  transportType?: TransportType;
  /** App-owned factory, sync or lazy. Remount to change it. */
  transportFactory?: TransportFactory;
  /** Read at construction; remount to change. */
  transportOptions?: TransportOptions;
  /** Construction options; microphone defaults on and camera off. */
  clientOptions?: Partial<Omit<PipecatClientOptions, "transport">>;
  /** Read per attempt; an APIRequest starts a bot before connecting. */
  connectParams?: TransportConnectionParams | APIRequest;
  /** Start request with optional response transformation; overrides connectParams. */
  startBotParams?: APIRequest;
  /** Transforms the startBot response before it is passed to connect(). */
  startBotResponseTransformer?: (
    response: TransportConnectionParams,
  ) => TransportConnectionParams | Promise<TransportConnectionParams>;
  /** Connect automatically once the client is ready. Default false. */
  connectOnMount?: boolean;
  /** Call client.initDevices() as soon as the client exists. Default false. */
  initDevicesOnMount?: boolean;
  /** Runs before device initialization or connection. */
  onClient?: (client: PipecatClient) => void;
}

export interface UsePipecatAppReturn {
  /** Null until the transport module has loaded and the client is built. */
  client: PipecatClient | null;
  /**
   * Starts the session (startBot flow when configured). No-op unless the
   * client state is initialized, disconnected, or error.
   */
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  /**
   * Failure message from transport loading, device init, or connecting —
   * null while healthy. Cleared automatically when connect() retries.
   */
  error: string | null;
  clearError: () => void;
  /** Raw client.startBot() response from the most recent connect. */
  rawStartBotResponse: unknown;
  /** The startBot response after `startBotResponseTransformer` ran. */
  transformedStartBotResponse: unknown;
}

/** Owns the client lifecycle. Only transportType rebuilds it; remount to change construction options. */
export function usePipecatApp(
  options: UsePipecatAppOptions = {},
): UsePipecatAppReturn {
  const { transportType = "smallwebrtc" } = options;

  // Read current options without recreating a live client.
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const activeClientRef = useRef<PipecatClient | null>(null);
  const connectingRef = useRef<PipecatClient | null>(null);
  const attemptRef = useRef<object | null>(null);

  const [client, setClient] = useState<PipecatClient | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rawStartBotResponse, setRawStartBotResponse] = useState<unknown>(null);
  const [transformedStartBotResponse, setTransformedStartBotResponse] =
    useState<unknown>(null);

  const startAndConnect = useCallback(async (activeClient: PipecatClient) => {
    if (
      activeClientRef.current !== activeClient ||
      connectingRef.current === activeClient
    )
      return;
    connectingRef.current = activeClient;
    const attempt = {};
    attemptRef.current = attempt;
    const isCurrent = () =>
      activeClientRef.current === activeClient &&
      attemptRef.current === attempt;

    const { connectParams, startBotParams, startBotResponseTransformer } =
      optionsRef.current;
    const activeTransportType =
      optionsRef.current.transportType ?? "smallwebrtc";
    try {
      // SDK connect() does not reject device-init failures; handle them here.
      if (activeClient.needsInit()) {
        await activeClient.initDevices();
        if (!isCurrent()) {
          if (
            activeClientRef.current !== activeClient ||
            attemptRef.current === null
          ) {
            await activeClient.disconnect();
          }
          return;
        }
      }
      if (startBotParams) {
        const response = await activeClient.startBot({
          requestData: {},
          ...startBotParams,
        });
        if (!isCurrent()) return;
        setRawStartBotResponse(response);
        if (
          activeTransportType === "smallwebrtc" &&
          typeof response === "object" &&
          response !== null &&
          "iceConfig" in response
        ) {
          const { iceConfig } = response as {
            iceConfig: { iceServers: RTCIceServer[] };
          };
          // Avoid a type import from an optional transport package.
          (
            activeClient.transport as { iceServers?: RTCIceServer[] }
          ).iceServers = iceConfig.iceServers;
        }
        const transformed = startBotResponseTransformer
          ? await startBotResponseTransformer(response)
          : response;
        if (!isCurrent()) return;
        await activeClient.connect(transformed);
        if (isCurrent()) setTransformedStartBotResponse(transformed);
      } else if (
        connectParams &&
        typeof connectParams === "object" &&
        "endpoint" in connectParams
      ) {
        // Check cancellation between bot startup and transport connection.
        const response = await activeClient.startBot(
          connectParams as APIRequest,
        );
        if (!isCurrent()) return;
        await activeClient.connect(response);
      } else {
        await activeClient.connect(connectParams ?? {});
      }
    } catch (err) {
      if (!isCurrent()) return;
      setError(
        `Failed to start session: ${err instanceof Error ? err.message : String(err)}`,
      );
      await activeClient.disconnect().catch(() => {});
    } finally {
      if (attemptRef.current === attempt) connectingRef.current = null;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    let createdClient: PipecatClient | null = null;

    (async () => {
      const { transportOptions, transportFactory, clientOptions } =
        optionsRef.current;
      let transport;
      try {
        transport = transportFactory
          ? await transportFactory(transportOptions)
          : await createTransport(transportType, transportOptions);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
        return;
      }
      if (cancelled) return;

      try {
        const pcClient = new PipecatClient({
          enableCam: false,
          enableMic: true,
          ...clientOptions,
          transport,
        });
        createdClient = pcClient;
        activeClientRef.current = pcClient;
        const setupAttempt = {};
        attemptRef.current = setupAttempt;
        setClient(pcClient);
        optionsRef.current.onClient?.(pcClient);

        if (optionsRef.current.initDevicesOnMount) {
          await pcClient.initDevices();
          if (cancelled || attemptRef.current !== setupAttempt) {
            if (cancelled || attemptRef.current === null) {
              await pcClient.disconnect();
            }
            return;
          }
        }
        if (
          !cancelled &&
          attemptRef.current === setupAttempt &&
          optionsRef.current.connectOnMount
        ) {
          await startAndConnect(pcClient);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          await createdClient?.disconnect().catch(() => {});
        }
      }
    })();

    return () => {
      cancelled = true;
      if (activeClientRef.current === createdClient) {
        activeClientRef.current = null;
        connectingRef.current = null;
        attemptRef.current = null;
      }
      // Cleanup cannot report to an unmounted caller, but must handle rejection.
      void createdClient?.disconnect().catch(() => {});
      setClient(null);
      setError(null);
      setRawStartBotResponse(null);
      setTransformedStartBotResponse(null);
    };
  }, [transportType, startAndConnect]);

  const connect = useCallback(async () => {
    if (!client || !CONNECTABLE_STATES.includes(client.state)) return;
    setError(null);
    await startAndConnect(client);
  }, [client, startAndConnect]);

  const disconnect = useCallback(async () => {
    if (!client) return;
    attemptRef.current = null;
    connectingRef.current = null;
    try {
      await client.disconnect();
    } catch (err) {
      if (activeClientRef.current !== client) return;
      setError(
        `Failed to disconnect: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }, [client]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    client,
    connect,
    disconnect,
    error,
    clearError,
    rawStartBotResponse,
    transformedStartBotResponse,
  };
}
