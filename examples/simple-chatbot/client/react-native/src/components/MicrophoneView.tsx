import React, { useState, useMemo } from 'react';
import { View, StyleSheet, LayoutChangeEvent, ViewStyle } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import Colors from '../theme/Colors';
import { useVoiceClient } from '../context/VoiceClientContext';

interface MicrophoneViewProps {
  style?: ViewStyle; // Optional additional styles for the button container
}

const MicrophoneView: React.FC<MicrophoneViewProps> = ({ style }) => {
  const { isMicEnabled, localAudioLevel: audioLevel } = useVoiceClient();
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const onLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
      setDimensions({ width, height });
  };

  const { width } = dimensions;

  const circleSize = useMemo(() => width * 0.9, [width]);
  const innerCircleSize = useMemo(() => width * 0.82, [width]);
  const audioCircleSize = useMemo(() => audioLevel * width * 0.95, [audioLevel, width]);

  return (
    <View style={[styles.container, style]} onLayout={onLayout}>
      <View
        style={[
          styles.outerCircle,
          { width: circleSize, height: circleSize, borderRadius: circleSize / 2 },
        ]}
      >
        <View
          style={[
            styles.innerCircle,
            {
              backgroundColor: !isMicEnabled ? Colors.disabledMic : Colors.backgroundCircle,
              width: innerCircleSize,
              height: innerCircleSize,
              borderRadius: innerCircleSize / 2,
            },
          ]}
        />

        {isMicEnabled && (
          <View
            style={[
              styles.audioCircle,
              {
                width: audioCircleSize,
                height: audioCircleSize,
                borderRadius: audioCircleSize / 2,
              },
            ]}
          />
        )}

        <MaterialIcons
          name={!isMicEnabled ? "mic-off" : "mic"}
          size={width * 0.2}
          color="white"
          style={styles.micIcon}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  } as ViewStyle,
  outerCircle: {
    borderWidth: 1,
    borderColor: Colors.buttonsBorder,
    justifyContent: 'center',
    alignItems: 'center',
  } as ViewStyle,
  innerCircle: {
    position: 'absolute',
  } as ViewStyle,
  audioCircle: {
    position: 'absolute',
    backgroundColor: Colors.micVolume,
    opacity: 0.5,
  } as ViewStyle,
  micIcon: {
    position: 'absolute',
  },
});

export default MicrophoneView;
