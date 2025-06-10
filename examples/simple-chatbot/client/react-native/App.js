// Disabling the logs from react-native-webrtc
import debug from 'debug';
debug.disable('rn-webrtc:*');

// Ignoring the warnings from react-native-background-timer while they don't fix this issue:
// https://github.com/ocetnik/react-native-background-timer/issues/366
import { LogBox } from 'react-native';
LogBox.ignoreLogs([
  "`new NativeEventEmitter()` was called with a non-null argument without the required `addListener` method.",
  "`new NativeEventEmitter()` was called with a non-null argument without the required `removeListeners` method."
]);

// Enable debug logs
/*window.localStorage = window.localStorage || {};
window.localStorage.debug = '*';
window.localStorage.getItem = (itemName) => {
  console.log('Requesting the localStorage item ', itemName);
  return window.localStorage[itemName];
};*/

export { default } from './src/App';
