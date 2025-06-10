import {
  View,
  StyleSheet,
  Text,
  Image,
  TouchableOpacity,
} from 'react-native';

import React from "react"

import { useVoiceClient } from '../context/VoiceClientContext';

import { Images } from '../theme/Assets';
import { MaterialIcons } from '@expo/vector-icons';

import WaveformView from '../components/WaveformView';
import MicrophoneView from '../components/MicrophoneView';
import { SafeAreaView } from 'react-native-safe-area-context';
import Colors from '../theme/Colors';
import CustomButton from '../theme/CustomButton';

const MeetingView: React.FC = () => {

  const { leave, toggleMicInput, toggleCamInput, timerCountDown } = useVoiceClient();

  const timerString = (count: number): string => {
    const hours = Math.floor(count / 3600);
    const minutes = Math.floor((count % 3600) / 60);
    const seconds = count % 60;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <View style={styles.header}>
          <Image source={Images.dailyBot} style={styles.botImage} />
          <View style={styles.timerContainer}>
            <MaterialIcons name="timelapse" size={24} color="black" />
            <Text style={styles.timerText}>{timerString(timerCountDown)}</Text>
          </View>
        </View>

        <View style={styles.mainPanel}>
          <WaveformView/>
          <View style={styles.bottomControls}>
            <TouchableOpacity onPress={toggleMicInput}>
              <MicrophoneView
                style={styles.microphone}
              />
            </TouchableOpacity>
          </View>
        </View>

        {/* Bottom Panel */}
        <View style={styles.bottomPanel}>
          <CustomButton
            title="End"
            iconName={"exit-to-app"}
            onPress={leave}
            backgroundColor={Colors.black}
          />
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    width: "100%",
    backgroundColor: Colors.backgroundApp,
  },
  container: {
    flex: 1,
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: 10,
  },
  botImage: {
    width: 48,
    height: 48,
  },
  timerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.timer,
    padding: 10,
    borderRadius: 12,
  },
  timerText: {
    color: 'black',
    fontWeight: '500',
    fontSize: 18,
    marginLeft: 5,
  },
  mainPanel: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  bottomControls: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    paddingBottom: 20,
  },
  microphone: {
    width: 160,
    height: 160,
  },
  camera: {
    width: 120,
    height: 120,
  },
  bottomPanel: {
    paddingVertical: 10,
  },
  endButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'black',
    borderRadius: 12,
    padding: 10,
  },
  endText: {
    marginLeft: 5,
    color: 'white',
  },
});

export default MeetingView;
