import React, { useEffect, useState } from 'react';
import { LayoutChangeEvent, StyleSheet, Text, View, ViewStyle } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import Colors from '../theme/Colors';
import { useVoiceClient } from '../context/VoiceClientContext';

const dotCount = 5;

const WaveformView: React.FC = () => {
  const [audioLevels, setAudioLevels] = useState(Array(dotCount).fill(0));
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const { currentState: voiceClientStatus, botReady: isBotReady, remoteAudioLevel: audioLevel } = useVoiceClient();

  const onLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setDimensions({ width, height });
  };

  useEffect(() => {
    setAudioLevels((prevLevels) => [...prevLevels.slice(1), audioLevel]);
  }, [audioLevel]);

  const { width, height } = dimensions;
  const circleSize = width * 0.9;
  const innerCircleSize = width * 0.82;
  const barWidth = (width * 0.5) / dotCount;

  return (
    <View style={styles.container} onLayout={onLayout}>
      <View style={[styles.outerCircle, { width: circleSize, height: circleSize, borderRadius: circleSize / 2 }]}>
        <View
          style={[
            styles.innerCircle,
            {
              backgroundColor: isBotReady ? Colors.backgroundCircle : Colors.backgroundCircleNotConnected,
              width: innerCircleSize,
              height: innerCircleSize,
              borderRadius: innerCircleSize / 2,
            },
          ]}
        >
          {isBotReady ? (
            audioLevel > 0 ? (
              <View style={[styles.waveformContainer, { width: width * 0.5, height: width * 0.5 }]}>
                {audioLevels.map((level, index) => (
                  <View
                    key={index}
                    style={[
                      styles.waveformBar,
                      {
                        width: barWidth - 10, // Subtract some margin
                        height: level * height,
                      },
                    ]}
                  />
                ))}
              </View>
            ) : (
              <View style={[styles.dotContainer, { width: width * 0.5, height: height * 0.5 }]}>
                {Array(dotCount)
                  .fill(0)
                  .map((_, index) => (
                    <View key={index} style={[styles.dot, { width: height * 0.1, height: height * 0.1 }]} />
                  ))}
              </View>
            )
          ) : (
            <View style={styles.notReadyContainer}>
              <MaterialIcons name="hourglass-empty" size={32} color="white" />
              <Text style={styles.voiceClientStatusText}>{voiceClientStatus}</Text>
            </View>
          )}
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
    width: "100%",
  } as ViewStyle,
  outerCircle: {
    borderWidth: 1,
    borderColor: 'gray',
    justifyContent: 'center',
    alignItems: 'center',
  } as ViewStyle,
  innerCircle: {
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  } as ViewStyle,
  waveformContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  } as ViewStyle,
  waveformBar: {
    backgroundColor: 'white',
    maxHeight: '100%',
    borderRadius: 12,
  } as ViewStyle,
  dotContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  } as ViewStyle,
  dot: {
    backgroundColor: 'white',
    borderRadius: 50,
  } as ViewStyle,
  notReadyContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  } as ViewStyle,
  voiceClientStatusText: {
    color: 'white',
    marginTop: 10,
    fontSize: 16,
    fontWeight: 'bold',
  } as ViewStyle,
});

export default WaveformView;
