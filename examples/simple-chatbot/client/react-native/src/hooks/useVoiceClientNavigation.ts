import { useEffect } from 'react';
import { useNavigation, NavigationProp } from '@react-navigation/native';
import { useVoiceClient } from '../context/VoiceClientContext';

export type RootStackParamList = {
  Meeting: undefined;
  Prejoin: undefined;
};

export const useVoiceClientNavigation = () => {
  const navigation = useNavigation<NavigationProp<RootStackParamList>>();
  const { inCall } = useVoiceClient();

  useEffect(() => {
    if (inCall) {
      navigation.navigate('Meeting');
    } else {
      navigation.navigate('Prejoin');
    }
  }, [inCall, navigation]);

};
