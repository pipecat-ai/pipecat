import {
  View,
  SafeAreaView,
  StyleSheet,
  Text,
  Button,
  TextInput,
} from "react-native";
import React, { useEffect, useState, useCallback } from "react";
import Daily, {
  DailyMediaView,
  DailyEventObjectParticipant,
} from "@daily-co/react-native-daily-js";

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#f7f9fa",
    width: "100%",
  },
  outCallContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  inCallContainer: {
    position: "absolute",
    width: "100%",
    height: "100%",
  },
  dailyMediaView: {
    flex: 1,
    aspectRatio: 9 / 16,
  },
  roomUrlInput: {
    borderRadius: 8,
    marginVertical: 8,
    padding: 12,
    fontStyle: "normal",
    fontWeight: "normal",
    borderWidth: 1,
    width: "100%",
  },
  infoView: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  controlButton: {
    flex: 1,
  },
});

const ROOM_URL_TEMPLATE = "https://filipi.daily.co/public";

export default function App() {
  const [videoTrack, setVideoTrack] = useState();
  const [callObject, setCallObject] = useState();
  const [inCall, setInCall] = useState(false);
  const [roomUrl, setRoomUrl] = useState(ROOM_URL_TEMPLATE);
  const [remoteParticipantCount, setRemoteParticipantCount] = useState(0);

  const handleNewParticipantsState = (event: DailyEventObjectParticipant) => {
    const participant = event.participant;
    // Early out as needed to avoid display the local participant's video
    if (participant.local) {
      return;
    }
    const videoTrack = participant.tracks.video;
    setVideoTrack(videoTrack.persistentTrack);
    // Set participant count minus the local participant
    setRemoteParticipantCount(callObject.participantCounts().present - 1);
  };

  const joinRoom = () => {
    console.log("Joining room");
    callObject.join({
      url: roomUrl,
    });
  };

  const leaveRoom = async () => {
    console.log("Leaving the room");
    await callObject.leave();
  };

  // Create the callObject and join the meeting
  useEffect(() => {
    const callObject = Daily.createCallObject();
    setCallObject(callObject);
    return () => {};
  }, []);

  //Add the listeners
  useEffect(() => {
    if (!callObject) {
      return;
    }
    callObject
      .on("joined-meeting", () => setInCall(true))
      .on("left-meeting", () => setInCall(false))
      .on("participant-joined", handleNewParticipantsState)
      .on("participant-updated", handleNewParticipantsState)
      .on("participant-left", handleNewParticipantsState);
    return () => {};
  }, [callObject]);

  return (
    <SafeAreaView style={styles.safeArea}>
      {inCall ? (
        <View style={styles.inCallContainer}>
          {remoteParticipantCount > 0 ? (
            <DailyMediaView
              videoTrack={videoTrack}
              mirror={false}
              objectFit="cover"
              style={styles.dailyMediaView}
            />
          ) : (
            <View style={styles.infoView}>
              <Text>No one else is in the call yet!</Text>
              <Text>Invite others to join the call using this link:</Text>
              <Text>{roomUrl}</Text>
            </View>
          )}
          <Button
            style={styles.controlButton}
            onPress={() => leaveRoom()}
            title="Leave call"
          ></Button>
        </View>
      ) : (
        <View style={styles.outCallContainer}>
          <View style={styles.infoView}>
            <Text>Not in a call yet</Text>
            <TextInput
              style={styles.roomUrlInput}
              value={roomUrl}
              onChangeText={(newRoomURL) => {
                setRoomUrl(newRoomURL);
              }}
            />
            <Button
              style={styles.controlButton}
              onPress={() => joinRoom()}
              title="Join call"
            ></Button>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}
