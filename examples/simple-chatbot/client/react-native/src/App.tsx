import React from "react"

import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import PreJoinView from './views/PreJoinView';
import MeetingView from './views/MeetingView';
import { VoiceClientProvider } from './context/VoiceClientContext';
import Toast from 'react-native-toast-message';

import { useVoiceClientNavigation } from './hooks/useVoiceClientNavigation';

const Stack = createStackNavigator();

const NavigationManager: React.FC = () => {
  useVoiceClientNavigation();  // This hook now controls the navigation based on the connection state.
  return null; // This component doesn't render anything but manages navigation.
};

const App: React.FC = () => {
  return (
    <VoiceClientProvider>
      <NavigationContainer>
        <Stack.Navigator initialRouteName="Prejoin">
          <Stack.Screen name="Prejoin" component={PreJoinView} options={{ headerShown: false }}/>
          <Stack.Screen name="Meeting" component={MeetingView} options={{ headerShown: false }}/>
        </Stack.Navigator>
        <NavigationManager />
        <Toast />
      </NavigationContainer>
    </VoiceClientProvider>
  );
};

export default App;
