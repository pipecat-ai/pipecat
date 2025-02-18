import React, { useState, useEffect } from 'react';
import {SafeAreaView, View, Text, Button, StyleSheet, ScrollView} from 'react-native';
import Daily from "@daily-co/react-native-daily-js";
import { API_BASE_URL } from "@env";

const CallScreen = () => {
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [isConnected, setIsConnected] = useState(false);
  const [callObject, setCallObject] = useState(null);
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    if (callObject) {
      setupTrackListeners(callObject);
    }
  }, [callObject]);

  const log = (message) => {
    setLogs((prevLogs) => [...prevLogs, `${new Date().toISOString()} - ${message}`]);
    console.log(message);
  };

  const setupTrackListeners = (callObject) => {
    callObject.on("joined-meeting", () => {
      setConnectionStatus('Connected');
      setIsConnected(true);
      log('Client connected');
    });
    callObject.on("left-meeting", () => {
      setConnectionStatus('Disconnected');
      setIsConnected(false);
      log('Client disconnected');
    });
    callObject.on("participant-left", () => {
      // When the bot leaves, we are also disconnecting from the call
      disconnect().catch((err) => {
        log(`Failed to disconnect ${err}`);
      })
    });
    // Trigger so the bot can start sending audio
    callObject.on("track-started", (evt) => {
      if (evt.track.kind === "audio" && evt.participant.local === false) {
        handleEventToConsole(evt)
        log("Sending the message that will trigger the bot to play the audio.")
        callObject.sendAppMessage("playable")
      }
    });
    callObject.on("error", (evt) => log(`Error: ${evt.error}`));
    // Other events just for awareness
    callObject.on("track-stopped", handleEventToConsole);
    callObject.on("participant-joined", handleEventToConsole);
    callObject.on("participant-updated", handleEventToConsole);
  };

  const handleEventToConsole = (evt) => {
    log(`Received event: ${evt.action}`);
  };

  const connect = async () => {
    try {
      const callObject = Daily.createCallObject({ subscribeToTracksAutomatically: true });
      setCallObject(callObject);
      const connectionUrl = `${API_BASE_URL}/connect`
      const res = await fetch(connectionUrl, { method: "POST", headers: { "Content-Type": "application/json" } });
      const roomInfo = await res.json();
      await callObject.join({ url: roomInfo.room_url });
    } catch (error) {
      log(`Error connecting: ${error.message}`);
    }
  };

  const disconnect = async () => {
    if (callObject) {
      try {
        await callObject.leave();
        await callObject.destroy();
        setCallObject(null);
      } catch (error) {
        log(`Error disconnecting: ${error.message}`);
      }
    }
  };

  return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.container}>
          <View style={styles.statusBar}>
            <Text>Status: <Text style={styles.status}>{connectionStatus}</Text></Text>
            <View style={styles.controls}>
              <Button
                title={isConnected ? "Disconnect" : "Connect"}
                onPress={isConnected ? disconnect : connect}
              />
            </View>
          </View>

          <View style={styles.debugPanel}>
            <Text style={styles.debugTitle}>Debug Info</Text>
            <ScrollView style={styles.debugLog}>
              {logs.map((logEntry, index) => (
                  <Text key={index} style={styles.logText}>{logEntry}</Text>
              ))}
            </ScrollView>
          </View>
        </View>
      </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#f0f0f0', padding: 20 },
  container: { flex: 1, margin: 20 },
  statusBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 10, backgroundColor: '#fff', borderRadius: 8, marginBottom: 20 },
  status: { fontWeight: 'bold' },
  controls: { flexDirection: 'row', gap: 10 },
  debugPanel: { height: '80%', backgroundColor: '#fff', borderRadius: 8, padding: 20},
  debugTitle: { fontSize: 16, fontWeight: 'bold' },
  debugLog: { height: '100%', overflow: 'scroll', backgroundColor: '#f8f8f8', padding: 10, borderRadius: 4, fontFamily: 'monospace', fontSize: 12, lineHeight: 1.4 },
});

export default CallScreen;
