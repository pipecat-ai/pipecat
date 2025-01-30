import React, { useState, useEffect } from 'react';
import {SafeAreaView, View, Text, Button, StyleSheet, ScrollView} from 'react-native';
import Daily from "@daily-co/react-native-daily-js";

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
    callObject.on("error", (evt) => log(`Error: ${evt.error}`));
    // Other events just for awareness
    callObject.on("track-started", (evt) => {
      handleEventToConsole(evt)
      log("Will send the audio message to play the audio at the next tick")
      callObject.sendAppMessage("playable")
    });
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
      const connectionUrl = 'http://192.168.1.16:7860/connect'
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
              <Button title="Connect" onPress={connect} disabled={isConnected} />
              <Button title="Disconnect" onPress={disconnect} disabled={!isConnected} />
            </View>
          </View>

          <View style={styles.debugPanel}>
            <Text style={styles.debugTitle}>Debug Info</Text>
            <View style={styles.debugLog}>
              <ScrollView style={styles.debugLog}>
                {logs.map((logEntry, index) => (
                    <Text key={index} style={styles.logText}>{logEntry}</Text>
                ))}
              </ScrollView>
            </View>
          </View>
        </View>
      </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#f0f0f0', padding: 20 },
  container: { flex: 1, maxWidth: 1200, margin: 'auto' },
  statusBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 10, backgroundColor: '#fff', borderRadius: 8, marginBottom: 20 },
  status: { fontWeight: 'bold' },
  controls: { flexDirection: 'row', gap: 10 },
  debugPanel: { backgroundColor: '#fff', borderRadius: 8, padding: 20 },
  debugTitle: { fontSize: 16, fontWeight: 'bold' },
  debugLog: { height: 200, overflow: 'scroll', backgroundColor: '#f8f8f8', padding: 10, borderRadius: 4, fontFamily: 'monospace', fontSize: 12, lineHeight: 1.4 },
});

export default CallScreen;
