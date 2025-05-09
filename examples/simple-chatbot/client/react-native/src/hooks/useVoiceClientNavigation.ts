import { useEffect } from 'react';
import { useNavigation } from '@react-navigation/native';
import { useVoiceClient } from '../context/VoiceClientContext';

export const useVoiceClientNavigation = () => {
  const navigation = useNavigation();
  const { inCall } = useVoiceClient();

  useEffect(() => {
    if (inCall) {
      // TODO, refactor this
      // @ts-ignore
      navigation.navigate('Meeting');
    } else {
      // TODO, refactor this
      // @ts-ignore
      navigation.navigate('Prejoin');
    }
  }, [inCall, navigation]);

};
