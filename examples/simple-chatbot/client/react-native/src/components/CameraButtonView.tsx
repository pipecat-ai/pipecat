import { View, Image, StyleSheet, LayoutChangeEvent, ImageStyle, ViewStyle } from 'react-native';

import React, { useMemo, useState } from 'react';

import { Icons } from '../theme/Assets';
import Colors from '../theme/Colors';

import { useVoiceClient } from '../context/VoiceClientContext';
import { VoiceClientVideoView } from '@pipecat-ai/react-native-daily-transport';

interface CameraButtonViewProps {
  style?: ViewStyle; // Optional additional styles for the button container
}

const CameraButtonView: React.FC<CameraButtonViewProps> = ({ style }) => {
  const { videoTrack, isCamEnabled } = useVoiceClient();
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const onLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setDimensions({ width, height });
  };

  const mediaComponent = useMemo(() => {
    return (
      <VoiceClientVideoView
        videoTrack={videoTrack || null}
        audioTrack={null}
        mirror={true}
        zOrder={1}
        style={styles.media}
        objectFit="cover"
      />
    );
  }, [videoTrack]);

  const { width } = dimensions;
  const circleSize = width * 0.9;
  const innerCircleSize = width * 0.82;

  return (
    <View style={[styles.container, style]} onLayout={onLayout}>
      <View
        style={[
          styles.outerCircle,
          { width: circleSize, height: circleSize, borderRadius: circleSize / 2 },
        ]}
      >
        {isCamEnabled ? (
          <View style={[styles.videoView, { borderRadius: circleSize / 2 }]}>
            {mediaComponent}
          </View>
        ) : (
          <>
            <View
              style={[
                styles.innerCircle,
                {
                  width: innerCircleSize,
                  height: innerCircleSize,
                  borderRadius: innerCircleSize / 2,
                },
              ]}
            />
            <Image
              source={Icons.vision}
              style={[
                styles.image,
                {
                  width: width * 0.3,
                  height: width * 0.3,
                  tintColor: 'green',
                },
              ]}
              resizeMode="contain"
            />
          </>
        )}
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
    backgroundColor: Colors.disabledVision,
    position: 'absolute',
  } as ViewStyle,
  videoView: {
    aspectRatio: 1,
    width: '100%',
    height: '100%',
    overflow: 'hidden',
  } as ViewStyle,
  image: {} as ImageStyle,
  media: {
    width: '100%',
    height: '100%',
    position: 'absolute',
  },
});

export default CameraButtonView;
