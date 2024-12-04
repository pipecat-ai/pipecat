// Type definitions for daily-js
// Project: https://github.com/daily-co/daily-js
// Definitions by: Paul Kompfner <https://github.com/kompfner>

/**
 * --- BROWSER-SPECIFIC TYPES ---
 */

/// <reference lib="dom" />

/**
 * --- DAILY-JS API ---
 */

export type DailyLanguage =
  | 'da'
  | 'de'
  | 'en'
  | 'es'
  | 'fi'
  | 'fr'
  | 'it'
  | 'jp'
  | 'ka'
  | 'nl'
  | 'no'
  | 'pl'
  | 'pt'
  | 'pt-BR'
  | 'ru'
  | 'sv'
  | 'tr';

export type DailyLanguageSetting = DailyLanguage | 'user';

export type DailyEvent =
  | 'loading'
  | 'load-attempt-failed'
  | 'loaded'
  | 'started-camera'
  | 'camera-error'
  | 'joining-meeting'
  | 'joined-meeting'
  | 'left-meeting'
  | 'call-instance-destroyed'
  | 'participant-joined'
  | 'participant-updated'
  | 'participant-left'
  | 'participant-counts-updated'
  | 'track-started'
  | 'track-stopped'
  | 'recording-started'
  | 'recording-stopped'
  | 'recording-stats'
  | 'recording-error'
  | 'recording-upload-completed'
  | 'recording-data'
  | 'transcription-started'
  | 'transcription-stopped'
  | 'transcription-error'
  | 'app-message'
  | 'transcription-message'
  | 'local-screen-share-started'
  | 'local-screen-share-stopped'
  | 'local-screen-share-canceled'
  | 'active-speaker-change'
  | 'active-speaker-mode-change'
  | 'network-quality-change'
  | 'network-connection'
  | 'test-completed'
  | 'cpu-load-change'
  | 'face-counts-updated'
  | 'fullscreen'
  | 'exited-fullscreen'
  | 'error'
  | 'nonfatal-error'
  | 'click'
  | 'mousedown'
  | 'mouseup'
  | 'mouseover'
  | 'mousemove'
  | 'touchstart'
  | 'touchmove'
  | 'touchend'
  | 'live-streaming-started'
  | 'live-streaming-updated'
  | 'live-streaming-stopped'
  | 'live-streaming-error'
  | 'lang-updated'
  | 'remote-media-player-started'
  | 'remote-media-player-stopped'
  | 'remote-media-player-updated'
  | 'access-state-updated'
  | 'meeting-session-updated'
  | 'meeting-session-summary-updated'
  | 'meeting-session-state-updated'
  | 'waiting-participant-added'
  | 'waiting-participant-updated'
  | 'waiting-participant-removed'
  | 'theme-updated'
  | 'available-devices-updated'
  | 'receive-settings-updated'
  | 'input-settings-updated'
  | 'send-settings-updated'
  | 'local-audio-level'
  | 'remote-participants-audio-level'
  | 'show-local-video-changed'
  | 'selected-devices-updated'
  | 'custom-button-click'
  | 'sidebar-view-changed'
  | 'dialin-connected'
  | 'dialin-ready'
  | 'dialin-error'
  | 'dialin-stopped'
  | 'dialin-warning'
  | 'dialout-connected'
  | 'dialout-error'
  | 'dialout-stopped'
  | 'dialout-warning';

export type DailyMeetingState =
  | 'new'
  | 'loading'
  | 'loaded'
  | 'joining-meeting'
  | 'joined-meeting'
  | 'left-meeting'
  | 'error';

export type DailyCameraErrorType =
  | 'cam-in-use'
  | 'mic-in-use'
  | 'cam-mic-in-use'
  | 'permissions'
  | 'undefined-mediadevices'
  | 'not-found'
  | 'constraints'
  | 'unknown';

export type DailyFatalErrorType =
  | 'ejected'
  | 'nbf-room'
  | 'nbf-token'
  | 'exp-room'
  | 'exp-token'
  | 'no-room'
  | 'meeting-full'
  | 'end-of-life'
  | 'not-allowed'
  | 'connection-error';

export type DailyNonFatalErrorType =
  | 'input-settings-error'
  | 'screen-share-error'
  | 'local-audio-level-observer-error'
  | 'video-processor-error'
  | 'audio-processor-error'
  | 'remote-media-player-error'
  | 'live-streaming-warning'
  | 'meeting-session-data-error';

export type DailyNetworkTopology = 'sfu' | 'peer';

export interface DailyParticipantsObject {
  local: DailyParticipant;
  [id: string]: DailyParticipant;
}

export interface DailyBrowserInfo {
  supported: boolean;
  mobile: boolean;
  name: string;
  version: string;
  supportsFullscreen: boolean;
  supportsScreenShare: boolean;
  supportsSfu: boolean;
  supportsVideoProcessing: boolean;
  supportsAudioProcessing: boolean;
}

export interface DailyThemeColors {
  /**
   * Main theme color. Used for primary actions and keyboard focus.
   */
  accent?: string;
  /**
   * Text color rendered on `accent`.
   */
  accentText?: string;
  /**
   * Background color.
   */
  background?: string;
  /**
   * Background color for highlighted elements.
   */
  backgroundAccent?: string;
  /**
   * Default text color, as rendered on `background` or `backgroundAccent`.
   */
  baseText?: string;
  /**
   * Default border color for bordered elements.
   */
  border?: string;
  /**
   * Background color for the call main area.
   */
  mainAreaBg?: string;
  /**
   * Background color for video tiles.
   */
  mainAreaBgAccent?: string;
  /**
   * Text color for text rendered inside the call main area, e.g. names.
   */
  mainAreaText?: string;
  /**
   * Text color for supportive, less emphasized, text.
   */
  supportiveText?: string;
}

export type DailyTheme = {
  colors: DailyThemeColors;
};
export type DailyThemeConfig =
  | DailyTheme
  | {
      light: DailyTheme;
      dark: DailyTheme;
    };

export interface DailyGridLayoutConfig {
  maxTilesPerPage?: number;
  minTilesPerPage?: number;
}
export interface DailyLayoutConfig {
  grid?: DailyGridLayoutConfig;
}

export interface DailyCustomTrayButton {
  iconPath: string;
  iconPathDarkMode?: string;
  label: string;
  tooltip: string;
}

export interface DailyCustomTrayButtons {
  [id: string]: DailyCustomTrayButton;
}

export interface DailyCustomIntegration {
  /**
   * Specifies the feature policy for the iframe.
   * https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe#attr-allow
   */
  allow?: HTMLIFrameElement['allow'];
  /**
   * Specifies who in the call is able to start and stop this integration.
   * - '*' means all participants can start and stop this integration
   * - 'owners' means only meeting owners can start and stop
   * - string[] defines the list of participants identified by their session_id
   * Default: '*'
   */
  controlledBy?: '*' | 'owners' | string[];
  /**
   * Specifies the [Content Security Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP) for the iframe.
   * Please check browser support before using this property.
   * https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe#attr-csp
   * https://caniuse.com/mdn-api_htmliframeelement_csp
   */
  csp?: string;
  /**
   * Specifies a publicly available URL to an icon image file associated with the integration.
   */
  iconURL?: string;
  /**
   * Used to render the integration's name in Prebuilt.
   */
  label: string;
  /**
   * By default integrations will be loaded lazily.
   */
  loading?: 'eager' | 'lazy';
  /**
   * Daily supports two different types of custom integrations:
   * - Main call area integrations
   * - Sidebar integrations
   */
  location: 'main' | 'sidebar';
  /**
   * A unique name for the iframe.
   * For more info see https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe#attr-name
   */
  name?: HTMLIFrameElement['name'];
  /**
   * https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe#attr-referrerpolicy
   */
  referrerPolicy?: HTMLIFrameElement['referrerPolicy'];
  /**
   * https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe#attr-sandbox
   */
  sandbox?: string;
  /**
   * The iframe's source URL.
   * https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe#attr-src
   */
  src?: HTMLIFrameElement['src'];
  /**
   * Allows to integrate inline HTML in an iframe.
   * https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe#attr-srcdoc
   * https://caniuse.com/iframe-srcdoc
   */
  srcdoc?: HTMLIFrameElement['srcdoc'];
  /**
   * When configured, shares the integration's state with other participants in the call:
   * - true will share with all other participants
   * - false won't share
   * - 'owners' will share with owners only
   * - string[] will share with participants with given list of session ids
   *
   * When the integration is started, it will be started for other participants, too.
   * When it's stopped, it will stop for all participants.
   */
  shared?: string[] | 'owners' | boolean;
}

export interface DailyCustomIntegrations {
  [id: string]: DailyCustomIntegration;
}

export interface DailyCallOptions {
  url?: string;
  token?: string;
  lang?: DailyLanguageSetting;
  activeSpeakerMode?: boolean;
  showLeaveButton?: boolean;
  showParticipantsBar?: boolean;
  showLocalVideo?: boolean;
  showFullscreenButton?: boolean;
  showUserNameChangeUI?: boolean;
  iframeStyle?: Partial<CSSStyleDeclaration>;
  customIntegrations?: DailyCustomIntegrations;
  customLayout?: boolean;
  customTrayButtons?: DailyCustomTrayButtons;
  bodyClass?: string;
  cssFile?: string;
  cssText?: string;
  dailyConfig?: DailyAdvancedConfig;
  subscribeToTracksAutomatically?: boolean;
  videoSource?: string | MediaStreamTrack | boolean;
  audioSource?: string | MediaStreamTrack | boolean;
  theme?: DailyThemeConfig;
  layoutConfig?: DailyLayoutConfig;
  receiveSettings?: DailyReceiveSettings;
  sendSettings?: DailySendSettings;
  inputSettings?: DailyInputSettings;
  userName?: string;
  userData?: unknown;
  startVideoOff?: boolean;
  startAudioOff?: boolean;
}

export interface StartCustomTrackOptions {
  track: MediaStreamTrack;
  mode?: 'music' | 'speech' | DailyMicAudioModeSettings | undefined;
  trackName?: string;
}

export interface DailyLoadOptions extends DailyCallOptions {
  baseUrl?: string;
}

export interface DailyFactoryOptions extends DailyCallOptions {
  // only available at constructor time
  allowMultipleCallInstances?: boolean;
  strictMode?: boolean;
}

export interface DailyMicAudioModeSettings {
  bitrate?: number;
  stereo?: boolean;
}

export interface DailyIceConfig {
  iceServers?: RTCIceServer[];
  placement?: 'front' | 'back' | 'replace';
  iceTransportPolicy?: RTCIceTransportPolicy;
}

export interface DailyAdvancedConfig {
  /**
   * @deprecated This property will be removed. Instead, use sendSettings, which is found in DailyCallOptions.
   */
  camSimulcastEncodings?: any[];
  /**
   * @deprecated This property will be removed. Use the method updateSendSettings instead.
   */
  disableSimulcast?: boolean;
  keepCamIndicatorLightOn?: boolean;
  v2CamAndMic?: boolean;
  /**
   * @deprecated This property will be removed. It has no affect.
   */
  fastConnect?: boolean;
  h264Profile?: string;
  micAudioMode?: 'music' | 'speech' | DailyMicAudioModeSettings;
  noAutoDefaultDeviceChange?: boolean;
  preferH264?: boolean;
  preferH264ForCam?: boolean;
  preferH264ForScreenSharing?: boolean;
  /**
   * @deprecated This property will be removed. Instead, use sendSettings, which
   *             is found in DailyCallOptions.
   */
  screenSimulcastEncodings?: any[];
  useDevicePreferenceCookies?: boolean;
  userMediaAudioConstraints?: MediaTrackConstraints;
  userMediaVideoConstraints?: MediaTrackConstraints;
  avoidEval?: boolean;
  callObjectBundleUrlOverride?: string;
  alwaysIncludeMicInPermissionPrompt?: boolean;
  alwaysIncludeCamInPermissionPrompt?: boolean;
  enableIndependentDevicePermissionPrompts?: boolean;
  proxyUrl?: string;
  iceConfig?: DailyIceConfig;
  useLegacyVideoProcessor?: boolean;
}

export interface DailyTrackState {
  subscribed: DailyTrackSubscriptionState;
  state:
    | 'blocked'
    | 'off'
    | 'sendable'
    | 'loading'
    | 'interrupted'
    | 'playable';
  blocked?: {
    byDeviceMissing?: boolean;
    byDeviceInUse?: boolean;
    byPermissions?: boolean;
  };
  off?: {
    byUser?: boolean;
    byRemoteRequest?: boolean;
    byBandwidth?: boolean;
    byCanSendPermission?: boolean;
    byServerLimit?: boolean;
  };
  // guaranteed-playable reference to the track
  // (it's only present when state === 'playable')
  track?: MediaStreamTrack;
  // not-guaranteed-playable reference to the track
  // (it may be present when state !== 'playable')
  // useful, for instance, for avoiding Safari's
  // remote-track-unmute-in-background-tab bug
  // (see https://github.com/daily-demos/call-object-react/blob/c81b21262dead2aacbd5a2f534d0fee8530acfe4/src/components/Tile/Tile.js#L53-L60)
  persistentTrack?: MediaStreamTrack;
}

export type DailyParticipantPermissionsCanSendValues =
  | 'video'
  | 'audio'
  | 'screenVideo'
  | 'screenAudio'
  | 'customVideo'
  | 'customAudio';

export type DailyParticipantPermissionsCanAdminValues =
  | 'participants'
  | 'streaming'
  | 'transcription';

export interface DailyParticipantPermissions {
  hasPresence: boolean;
  canSend: Set<DailyParticipantPermissionsCanSendValues> | boolean;
  canAdmin: Set<DailyParticipantPermissionsCanAdminValues> | boolean;
}

export type DailyParticipantPermissionsUpdate = {
  hasPresence?: boolean;
  canSend?:
    | Array<DailyParticipantPermissionsCanSendValues>
    | Set<DailyParticipantPermissionsCanSendValues>
    | boolean;
  canAdmin?:
    | Array<DailyParticipantPermissionsCanAdminValues>
    | Set<DailyParticipantPermissionsCanAdminValues>
    | boolean;
};

export interface DailyParticipantTracks {
  audio: DailyTrackState;
  video: DailyTrackState;
  screenAudio: DailyTrackState;
  screenVideo: DailyTrackState;
  rmpAudio?: DailyTrackState;
  rmpVideo?: DailyTrackState;
  [customTrackKey: string]: DailyTrackState | undefined;
}

export interface DailyParticipant {
  /**
   * @deprecated
   * This property will be removed. Use tracks.audio.persistentTrack instead.
   */
  audioTrack?: MediaStreamTrack | false;
  /**
   * @deprecated
   * This property will be removed. Use tracks.video.persistentTrack instead.
   */
  videoTrack?: MediaStreamTrack | false;
  /**
   * @deprecated
   * This property will be removed.
   * Use tracks.screenVideo.persistentTrack instead.
   */
  screenVideoTrack?: MediaStreamTrack | false;
  /**
   * @deprecated
   * This property will be removed.
   * Use tracks.screenAudio.persistentTrack instead.
   */
  screenAudioTrack?: MediaStreamTrack | false;

  /**
   * @deprecated This property will be removed. Use tracks.audio.state instead.
   */
  audio: boolean;
  /**
   * @deprecated This property will be removed. Use tracks.video.state instead.
   */
  video: boolean;
  /**
   * @deprecated
   * This property will be removed. Use tracks.screenVideo.state instead.
   */
  screen: boolean;

  // track state
  tracks: DailyParticipantTracks;

  // user/session info
  user_id: string;
  user_name: string;
  userData?: unknown;
  session_id: string;
  joined_at?: Date;
  networkThreshold?: 'good' | 'low' | 'very-low';
  will_eject_at: Date;
  local: boolean;
  owner: boolean;
  permissions: DailyParticipantPermissions;
  record: boolean;
  participantType?: string;

  // video element info (iframe-based calls using standard UI only)
  /**
   * @deprecated
   * This property will be removed. Refer to tracks.video instead.
   */
  cam_info: {} | DailyVideoElementInfo;
  /**
   * @deprecated
   * This property will be removed. Refer to tracks.screenVideo instead.
   */
  screen_info: {} | DailyVideoElementInfo;
}

export interface DailyParticipantCounts {
  present: number;
  hidden: number;
}

export interface DailyWaitingParticipant {
  id: string;
  name: string;
  awaitingAccess: SpecifiedDailyAccess;
}

export type DailyTrackSubscriptionState = 'staged' | boolean;

export type DailyCustomTrackSubscriptionState =
  | DailyTrackSubscriptionState
  | { [name: string]: DailyTrackSubscriptionState };

export type DailyTrackSubscriptionOptions =
  | DailyTrackSubscriptionState
  | 'avatar'
  | {
      audio?: DailyTrackSubscriptionState;
      video?: DailyTrackSubscriptionState;
      screenVideo?: DailyTrackSubscriptionState;
      screenAudio?: DailyTrackSubscriptionState;
      custom?: DailyCustomTrackSubscriptionState;
    };

export interface DailyParticipantUpdateOptions {
  setAudio?: boolean;
  setVideo?: boolean;
  setScreenShare?: false;
  setSubscribedTracks?: DailyTrackSubscriptionOptions;
  eject?: true;
  updatePermissions?: DailyParticipantPermissionsUpdate;
  styles?: DailyParticipantCss;
}

export interface DailyWaitingParticipantUpdateOptions {
  grantRequestedAccess?: boolean;
}

export interface DailyParticipantCss {
  cam?: DailyParticipantStreamCss;
  screen?: DailyParticipantStreamCss;
}

export interface DailyParticipantStreamCss {
  div?: Partial<CSSStyleDeclaration>;
  overlay?: Partial<CSSStyleDeclaration>;
  video?: Partial<CSSStyleDeclaration>;
}

/**
 * @deprecated
 * All properties will be removed as cam_info and screen_info are also
 * deprecated. Use the participants() object's tracks property to retrieve track
 * information instead.
 * e.g. call.participants()['participant-id'].tracks.video.persistentTrack.getSettings()
 */
export interface DailyVideoElementInfo {
  width: number;
  height: number;
  left: number;
  top: number;
  video_width: number;
  video_height: number;
}

export interface DailyDeviceInfos {
  camera: {} | DailyMediaDeviceInfo;
  mic: {} | MediaDeviceInfo;
  speaker: {} | MediaDeviceInfo;
}

/**
 * @deprecated
 * Almost all the properties in this type were just used by Electron.
 * And the mediaStream can be replaced to use custom tracks.
 */
export interface DailyScreenCaptureOptions {
  /**
   * @deprecated This property will be removed. It is only used for Electron.
   */
  audio?: boolean;
  /**
   * @deprecated This property will be removed. It is only used for Electron.
   */
  maxWidth?: number;
  /**
   * @deprecated This property will be removed. It is only used for Electron.
   */
  maxHeight?: number;
  /**
   * @deprecated This property will be removed. It is only used for Electron.
   */
  chromeMediaSourceId?: string;
  /**
   * @deprecated
   * This property will be removed.
   * It is recommended to use our custom tracks API.
   */
  mediaStream?: MediaStream;
}

// More details about all the possible options
// https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia
export interface DailyDisplayMediaStreamOptions {
  audio?: boolean | MediaTrackConstraints;
  video?: boolean | MediaTrackConstraints;
  selfBrowserSurface?: 'include' | 'exclude';
  surfaceSwitching?: 'include' | 'exclude';
  systemAudio?: 'include' | 'exclude';
}

export interface DailyDisplayMediaStreamOptionsElectron {
  audio?: boolean;
  video: {
    maxWidth?: number;
    maxHeight?: number;
  };
  chromeMediaSourceId?: string;
}

export interface DailyStartScreenShare {
  displayMediaOptions?:
    | DailyDisplayMediaStreamOptions
    | DailyDisplayMediaStreamOptionsElectron;
  screenVideoSendSettings?:
    | DailyVideoSendSettings
    | DailyScreenVideoSendSettingsPreset;
}

export interface DailyStartScreenShareFromStream {
  mediaStream: MediaStream;
  screenVideoSendSettings?:
    | DailyVideoSendSettings
    | DailyScreenVideoSendSettingsPreset;
}

export type DailyStartScreenShareOptions =
  | DailyScreenCaptureOptions
  | DailyStartScreenShare
  | DailyStartScreenShareFromStream;

export type DailyQualityTestResult =
  | 'good'
  | 'bad'
  | 'warning'
  | 'aborted'
  | 'failed';

export type DailyCallQualityTestResults =
  | DailyCallQualityTestStats
  | DailyCallQualityTestAborted
  | DailyCallQualityTestFailure;

export type DailyP2PCallQualityTestResults =
  | DailyP2PCallQualityTestStats
  | DailyCallQualityTestAborted
  | DailyCallQualityTestFailure;

export interface DailyCallQualityTestStats {
  result: Extract<DailyQualityTestResult, 'good' | 'warning' | 'bad'>;
  data: DailyCallQualityTestData;
  secondsElapsed: number;
}
export interface DailyP2PCallQualityTestStats {
  result: Extract<DailyQualityTestResult, 'good' | 'warning' | 'bad'>;
  data: DailyP2PCallQualityTestData;
  secondsElapsed: number;
}

export interface DailyCallQualityTestData {
  maxRoundTripTime: number | null;
  avgRoundTripTime: number | null;
  avgSendPacketLoss: number | null;
  avgAvailableOutgoingBitrate: number | null;
  avgSendBitsPerSecond: number | null;
}

export interface DailyP2PCallQualityTestData {
  maxRoundTripTime: number | null;
  avgRoundTripTime: number | null;
  avgRecvPacketLoss: number | null;
  avgAvailableOutgoingBitrate: number | null;
  avgSendBitsPerSecond: number | null;
  avgRecvBitsPerSecond: number | null;
}

export interface DailyCallQualityTestAborted {
  result: Extract<DailyQualityTestResult, 'aborted'>;
  secondsElapsed: number;
}

export interface DailyCallQualityTestFailure {
  result: Extract<DailyQualityTestResult, 'failed'>;
  errorMsg: string;
  error?: DailyFatalErrorObject<DailyFatalErrorType>;
  secondsElapsed: number;
}

export interface DailyConnectionQualityTestData {
  // TODO: New TestPeerToPeerCallQuality() should return DailyCallQualityTestData
  maxRTT: number | null;
  packetLoss: number | null;
}

export interface DailyConnectionQualityTestStats {
  result: DailyQualityTestResult;
  data: DailyConnectionQualityTestData;
  secondsElapsed: number;
}

export interface DailyWebsocketConnectivityTestResults {
  result: 'passed' | 'failed' | 'warning' | 'aborted';
  abortedRegions: string[];
  failedRegions: string[];
  passedRegions: string[];
}

export interface DailyNetworkConnectivityTestStats {
  result: 'passed' | 'failed' | 'aborted';
}

export interface DailyNetworkStats {
  quality: number;
  stats: {
    latest: {
      timestamp: number;
      recvBitsPerSecond: number | null;
      sendBitsPerSecond: number | null;
      availableOutgoingBitrate: number | null;
      networkRoundTripTime: number | null;
      videoRecvBitsPerSecond: number | null;
      videoSendBitsPerSecond: number | null;
      audioRecvBitsPerSecond: number | null;
      audioSendBitsPerSecond: number | null;
      videoRecvPacketLoss: number | null;
      videoSendPacketLoss: number | null;
      audioRecvPacketLoss: number | null;
      audioSendPacketLoss: number | null;
      totalSendPacketLoss: number | null;
      totalRecvPacketLoss: number | null;
      videoRecvJitter: number | null;
      videoSendJitter: number | null;
      audioRecvJitter: number | null;
      audioSendJitter: number | null;
    };
    worstVideoRecvPacketLoss: number;
    worstVideoSendPacketLoss: number;
    worstAudioRecvPacketLoss: number;
    worstAudioSendPacketLoss: number;
    worstVideoRecvJitter: number;
    worstVideoSendJitter: number;
    worstAudioRecvJitter: number;
    worstAudioSendJitter: number;
    averageNetworkRoundTripTime: number;
  };
  threshold: 'good' | 'low' | 'very-low';
}

export interface DailyCpuLoadStats {
  cpuLoadState: 'low' | 'high';
  cpuLoadStateReason: 'encode' | 'decode' | 'scheduleDuration' | 'none'; // We are currently not using the Inter frame Delay to change the cpu load state
  stats: {
    latest: {
      timestamp: number;
      scheduleDuration: number;
      frameEncodeTimeSec: number;
      targetEncodeFrameRate: number;
      targetDecodeFrameRate: number;
      targetScheduleDuration: number;
      cpuUsageBasedOnTargetEncode: number;
      cpuUsageBasedOnGlobalDecode: number;
      avgFrameDecodeTimeSec: number;
      avgInterFrameDelayStandardDeviation: number;
      totalReceivedVideoTracks: number;
      cpuInboundVideoStats: {
        trackId: string;
        ssrc: number;
        frameWidth: number;
        frameHeight: number;
        fps: number;
        frameDecodeTimeSec: number;
        interFrameDelayStandardDeviation: number;
        cpuUsageBasedOnTargetDecode: number;
      }[];
    };
  };
}

export interface DailySendSettings {
  video?: DailyCamVideoSendSettings | DailyVideoSendSettingsPreset;
  customVideoDefaults?: DailyVideoSendSettings | DailyVideoSendSettingsPreset;
  screenVideo?: DailyVideoSendSettings | DailyScreenVideoSendSettingsPreset;
  [customKey: string]:
    | DailyVideoSendSettings
    // TypeScript will prioritize the index signature over explicitly declared properties
    // So unless I add it here, in order to use DailyCamVideoSendSettings I would need to
    // use of a type assertion to inform TypeScript about the specific type of video.
    // Like this:
    // video: { allowAdaptiveLayers: true, } as DailyCamVideoSendSettings
    | DailyCamVideoSendSettings
    | DailyVideoSendSettingsPreset
    | DailyScreenVideoSendSettingsPreset
    | undefined;
}

export interface DailyParticipantsAudioLevel {
  [participantId: string]: number;
}

export type DailyVideoSendSettingsPreset =
  | 'default-video'
  | 'bandwidth-optimized'
  | 'bandwidth-and-quality-balanced'
  | 'quality-optimized'
  | 'adaptive-2-layers'
  | 'adaptive-3-layers';

// Media Track Send Settings
export interface DailyVideoSendSettings {
  maxQuality?: 'low' | 'medium' | 'high';
  encodings?: {
    low: RTCRtpEncodingParameters;
    medium?: RTCRtpEncodingParameters;
    high?: RTCRtpEncodingParameters;
  };
}

export interface DailyCamVideoSendSettings extends DailyVideoSendSettings {
  allowAdaptiveLayers?: boolean;
}

export type DailyScreenVideoSendSettingsPreset =
  | 'default-screen-video'
  | 'detail-optimized'
  | 'motion-optimized'
  | 'motion-and-detail-balanced';

export interface DailyPendingRoomInfo {
  roomUrlPendingJoin: string;
}

export interface DailyRecordingsBucket {
  allow_api_access: boolean;
  allow_streaming_from_bucket: boolean;
  assume_role_arn: string;
  bucket_name: string;
  bucket_region: string;
}

export interface DailyRoomInfo {
  id: string;
  name: string;
  config: {
    nbf?: number;
    exp?: number;
    max_participants?: number;
    enable_screenshare?: boolean;
    enable_advanced_chat?: boolean;
    enable_breakout_rooms?: boolean;
    enable_emoji_reactions?: boolean;
    enable_chat?: boolean;
    enable_shared_chat_history?: boolean;
    enable_hand_raising?: boolean;
    enable_knocking?: boolean;
    enable_live_captions_ui?: boolean;
    enable_network_ui?: boolean;
    enable_noise_cancellation_ui?: boolean;
    enable_people_ui?: boolean;
    enable_pip_ui?: boolean;
    enable_prejoin_ui?: boolean;
    enable_video_processing_ui?: boolean;
    experimental_optimize_large_calls?: boolean;
    start_video_off?: boolean;
    start_audio_off?: boolean;
    owner_only_broadcast?: boolean;
    audio_only?: boolean;
    enable_recording?: string;
    enable_dialin?: boolean;
    /**
     * @deprecated This property will be removed.
     * All calls are treated as autojoin.
     */
    autojoin?: boolean;
    eject_at_room_exp?: boolean;
    eject_after_elapsed?: number;
    lang?: '' | DailyLanguageSetting;
    sfu_switchover?: number;
    /**
     * @deprecated This property will be removed.
     * All calls use websocket signaling ('ws').
     */
    signaling_impl?: string;
    geo?: string;
    recordings_bucket?: DailyRecordingsBucket;
  };
  domainConfig: {
    hide_daily_branding?: boolean;
    redirect_on_meeting_exit?: string;
    hipaa?: boolean;
    sfu_impl?: string;
    signaling_impl?: string;
    sfu_switchover?: number;
    lang?: '' | DailyLanguageSetting;
    max_api_rooms?: number;
    webhook_meeting_end?: any;
    max_live_streams?: number;
    max_streaming_instances_per_room?: number;
    enable_advanced_chat?: boolean;
    enable_breakout_rooms?: boolean;
    enable_emoji_reactions?: boolean;
    enable_chat?: boolean;
    enable_shared_chat_history?: boolean;
    enable_hand_raising?: boolean;
    enable_live_captions_ui?: boolean;
    enable_network_ui?: boolean;
    enable_noise_cancellation_ui?: boolean;
    enable_people_ui?: boolean;
    enable_pip_ui?: boolean;
    enable_prejoin_ui?: boolean;
    enable_transcription?: boolean;
    enable_video_processing_ui?: boolean;
    recordings_bucket?: DailyRecordingsBucket;
  };
  tokenConfig: {
    eject_at_token_exp?: boolean;
    eject_after_elapsed?: number;
    nbf?: number;
    exp?: number;
    is_owner?: boolean;
    user_name?: string;
    user_id?: string;
    enable_live_captions_ui?: boolean;
    enable_prejoin_ui?: boolean;
    enable_screenshare?: boolean;
    start_video_off?: boolean;
    start_audio_off?: boolean;
    enable_recording?: string;
    start_cloud_recording_opts?: DailyStreamingOptions<'recording', 'start'>;
    enable_recording_ui?: boolean;
    start_cloud_recording?: boolean;
    close_tab_on_exit?: boolean;
    redirect_on_meeting_exit?: string;
    lang?: '' | DailyLanguageSetting;
  };
  dialInPIN?: string;
}

/**
 * @deprecated
 * This type will be removed. Use DailyMeetingSessionSummary instead.
 */
export interface DailyMeetingSession {
  id: string;
}

export interface DailyMeetingSessionSummary {
  id: string;
}

export interface DailyMeetingSessionState {
  data: unknown;
  topology: DailyNetworkTopology | 'none';
}

export type DailySessionDataMergeStrategy = 'replace' | 'shallow-merge';

export interface DailyVideoReceiveSettings {
  layer?: number;
}
export interface DailySingleParticipantReceiveSettings {
  video?: DailyVideoReceiveSettings;
  screenVideo?: DailyVideoReceiveSettings;
  [customKey: string]: DailyVideoReceiveSettings | undefined;
}

export interface DailyReceiveSettings {
  [participantIdOrBase: string]: DailySingleParticipantReceiveSettings;
}

export interface DailyVideoReceiveSettingsUpdates {
  layer?: number | 'inherit';
}

export interface DailySingleParticipantReceiveSettingsUpdates {
  video?: DailyVideoReceiveSettingsUpdates | 'inherit';
  screenVideo?: DailyVideoReceiveSettingsUpdates | 'inherit';
  [customKey: string]: DailyVideoReceiveSettingsUpdates | 'inherit' | undefined;
}

export interface DailyReceiveSettingsUpdates {
  [participantIdOrBaseOrStar: string]:
    | DailySingleParticipantReceiveSettingsUpdates
    | 'inherit';
}

export interface DailyInputSettings {
  audio?: DailyInputAudioSettings;
  video?: DailyInputVideoSettings;
}

export interface DailyInputAudioSettings {
  processor: DailyInputAudioProcessorSettings;
}

export interface DailyInputAudioProcessorSettings {
  type: 'none' | 'noise-cancellation';
}

export interface DailyNoInputSettings {
  type: 'none';
}

export interface DailyBackgroundBlurInputSettings {
  type: 'background-blur';
  config: {
    strength?: number;
  };
}

export interface DailyFaceDetectionInputSettings {
  type: 'face-detection';
}

export interface DailyBackgroundImageInputSettings {
  type: 'background-image';
  config: {
    url?: string;
    source?: string | number | ArrayBuffer;
  };
}

export type DailyInputVideoProcessorSettings =
  | DailyNoInputSettings
  | DailyBackgroundBlurInputSettings
  | DailyBackgroundImageInputSettings
  | DailyFaceDetectionInputSettings;

export interface DailyInputVideoSettings {
  processor?: DailyInputVideoProcessorSettings;
}

export type DailyEventObjectBase = {
  action: DailyEvent;
  callClientId: string;
};

export interface DailyEventObjectNoPayload extends DailyEventObjectBase {
  action: Extract<
    DailyEvent,
    | 'loading'
    | 'loaded'
    | 'joining-meeting'
    | 'left-meeting'
    | 'call-instance-destroyed'
    | 'recording-stats'
    | 'recording-upload-completed'
    | 'fullscreen'
    | 'exited-fullscreen'
  >;
}

export type DailyCameraError = {
  msg: string;
};

export interface DailyCamPermissionsError extends DailyCameraError {
  type: Extract<DailyCameraErrorType, 'permissions'>;
  blockedBy: 'user' | 'browser';
  blockedMedia: Array<'video' | 'audio'>;
}

export interface DailyCamDeviceNotFoundError extends DailyCameraError {
  type: Extract<DailyCameraErrorType, 'not-found'>;
  missingMedia: Array<'video' | 'audio'>;
}

export interface DailyCamConstraintsError extends DailyCameraError {
  type: Extract<DailyCameraErrorType, 'constraints'>;
  reason: 'invalid' | 'none-specified';
}

export interface DailyCamInUseError extends DailyCameraError {
  type: Extract<
    DailyCameraErrorType,
    'cam-in-use' | 'mic-in-use' | 'cam-mic-in-use'
  >;
}

export interface DailyCamTypeError extends DailyCameraError {
  type: Extract<DailyCameraErrorType, 'undefined-mediadevices'>;
}

export interface DailyCamUnknownError extends DailyCameraError {
  type: Extract<DailyCameraErrorType, 'unknown'>;
}

export type DailyCameraErrorObject<T extends DailyCameraErrorType> =
  T extends DailyCamPermissionsError['type']
    ? DailyCamPermissionsError
    : T extends DailyCamDeviceNotFoundError['type']
    ? DailyCamDeviceNotFoundError
    : T extends DailyCamConstraintsError['type']
    ? DailyCamConstraintsError
    : T extends DailyCamInUseError['type']
    ? DailyCamInUseError
    : T extends DailyCamTypeError['type']
    ? DailyCamTypeError
    : T extends DailyCamUnknownError['type']
    ? DailyCamUnknownError
    : any;

export interface DailyEventObjectCameraError extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'camera-error'>;
  errorMsg: {
    errorMsg: string;
    audioOk?: boolean;
    videoOk?: boolean;
  };
  error: DailyCameraErrorObject<DailyCameraErrorType>;
}

export type DailyFatalError = {
  type: DailyFatalErrorType;
  msg: string;
};

export interface DailyFatalConnectionError extends DailyFatalError {
  type: Extract<DailyFatalErrorType, 'connection-error'>;
  details: {
    on:
      | 'load'
      | 'join'
      | 'reconnect'
      | 'move'
      | 'rtc-connection'
      | 'room-lookup';
    sourceError: Record<string, any>;
    uri?: string;
    workerGroup?: string;
    geoGroup?: string;
    bundleUrl?: string;
  };
}

export type DailyFatalErrorObject<T extends DailyFatalErrorType> =
  T extends DailyFatalConnectionError['type'] ? DailyFatalConnectionError : any;

export interface DailyEventObjectFatalError extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'error'>;
  errorMsg: string;
  error?: DailyFatalErrorObject<DailyFatalErrorType>;
}

export interface DailyEventObjectNonFatalError extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'nonfatal-error'>;
  type: DailyNonFatalErrorType;
  errorMsg: string;
  details?: any;
}

export interface DailyEventObjectGenericError extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'load-attempt-failed'>;
  errorMsg: string;
}

export interface DailyEventObjectLiveStreamingError
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'live-streaming-error'>;
  errorMsg: string;
  instanceId?: string;
  actionTraceId?: string;
}

export interface DailyEventObjectParticipants extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'joined-meeting'>;
  participants: DailyParticipantsObject;
}

export interface DailyEventObjectParticipant extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'participant-joined' | 'participant-updated'>;
  participant: DailyParticipant;
}

// only 1 reason reported for now. more to come.
export type DailyParticipantLeftReason = 'hidden';

export interface DailyEventObjectParticipantLeft extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'participant-left'>;
  participant: DailyParticipant;
  // reason undefined if participant left for any reason other than those listed
  // in DailyParticipantLeftReason
  reason?: DailyParticipantLeftReason;
}

export interface DailyEventObjectParticipantCounts
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'participant-counts-updated'>;
  participantCounts: DailyParticipantCounts;
}

export interface DailyEventObjectWaitingParticipant
  extends DailyEventObjectBase {
  action: Extract<
    DailyEvent,
    | 'waiting-participant-added'
    | 'waiting-participant-updated'
    | 'waiting-participant-removed'
  >;
  participant: DailyWaitingParticipant;
}

export interface DailyEventObjectAccessState
  extends DailyAccessState,
    DailyEventObjectBase {
  action: Extract<DailyEvent, 'access-state-updated'>;
}

export interface DailyEventObjectMeetingSessionSummaryUpdated
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'meeting-session-summary-updated'>;
  meetingSession: DailyMeetingSessionSummary;
}

/**
 * @deprecated
 * This event will be removed. Use the
 * DailyEventObjectMeetingSessionSummaryUpdated type instead.
 */
export interface DailyEventObjectMeetingSessionUpdated
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'meeting-session-updated'>;
  meetingSession: DailyMeetingSession;
}

export interface DailyEventObjectMeetingSessionStateUpdated
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'meeting-session-state-updated'>;
  meetingSessionState: DailyMeetingSessionState;
}

export interface DailyEventObjectTrack extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'track-started' | 'track-stopped'>;
  participant: DailyParticipant | null; // null if participant left meeting
  track: MediaStreamTrack;
  type:
    | 'video'
    | 'audio'
    | 'screenVideo'
    | 'screenAudio'
    | 'rmpVideo'
    | 'rmpAudio'
    | string; // string - for custom tracks
}

export interface DailyEventObjectRecordingStarted extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'recording-started'>;
  local?: boolean;
  recordingId?: string;
  startedBy?: string;
  type?: string;
  layout?: DailyStreamingLayoutConfig<'start'>;
  instanceId?: string;
}

export interface DailyEventObjectRecordingStopped extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'recording-stopped'>;
  instanceId?: string;
}

export interface DailyEventObjectRecordingError extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'recording-error'>;
  errorMsg: string;
  instanceId?: string;
  actionTraceId?: string;
}

export interface DailyEventObjectRecordingData extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'recording-data'>;
  data: Uint8Array;
  finished: boolean;
}

export interface DailyEventObjectMouseEvent extends DailyEventObjectBase {
  action: Extract<
    DailyEvent,
    'click' | 'mousedown' | 'mouseup' | 'mouseover' | 'mousemove'
  >;
  event: {
    type: string;
    button: number;
    x: number;
    y: number;
    pageX: number;
    pageY: number;
    screenX: number;
    screenY: number;
    offsetX: number;
    offsetY: number;
    altKey: boolean;
    ctrlKey: boolean;
    metaKey: boolean;
    shiftKey: boolean;
  };
  participant: DailyParticipant;
}

export interface DailyEventObjectTouchEvent extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'touchstart' | 'touchmove' | 'touchend'>;
  event: {
    type: string;
    altKey: boolean;
    ctrlKey: boolean;
    metaKey: boolean;
    shiftKey: boolean;
  };
  participant: DailyParticipant;
}

export interface DailyEventObjectNetworkQualityEvent
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'network-quality-change'>;
  threshold: 'good' | 'low' | 'very-low';
  quality: number;
}

export interface DailyEventObjectCpuLoadEvent extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'cpu-load-change'>;
  cpuLoadState: 'low' | 'high';
  cpuLoadStateReason: 'encode' | 'decode' | 'scheduleDuration' | 'none'; // We are currently not using the Inter frame Delay to change the cpu load state
}

export interface DailyEventObjectFaceCounts extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'face-counts-updated'>;
  faceCounts: number;
}

export type NetworkConnectionType = 'signaling' | 'peer-to-peer' | 'sfu';

export interface DailyEventObjectNetworkConnectionEvent
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'network-connection'>;
  type: NetworkConnectionType;
  event: string;
  session_id?: string;
  sfu_id?: string;
}

export interface DailyEventObjectTestCompleted extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'test-completed'>;
  test:
    | 'call-quality'
    | 'p2p-call-quality'
    | 'network-connectivity'
    | 'websocket-connectivity';
  results:
    | DailyCallQualityTestResults
    | DailyP2PCallQualityTestResults
    | DailyNetworkConnectivityTestStats
    | DailyWebsocketConnectivityTestResults;
}

export interface DailyEventObjectActiveSpeakerChange
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'active-speaker-change'>;
  activeSpeaker: {
    peerId: string;
  };
}

export interface DailyEventObjectActiveSpeakerModeChange
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'active-speaker-mode-change'>;
  enabled: boolean;
}

export interface DailyEventObjectAppMessage<Data = any>
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'app-message'>;
  data: Data;
  fromId: string;
}

export interface DailyEventObjectTranscriptionMessage
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'transcription-message'>;
  instanceId?: string;
  participantId: string;
  text: string;
  timestamp: Date;
  rawResponse: Record<string, any>;
}

export interface DailyEventObjectLangUpdated extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'lang-updated'>;
  lang: DailyLanguage;
  langSetting: DailyLanguageSetting;
}

export interface DailyEventObjectThemeUpdated extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'theme-updated'>;
  theme: DailyThemeConfig;
}

export interface DailyEventObjectReceiveSettingsUpdated
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'receive-settings-updated'>;
  receiveSettings: DailyReceiveSettings;
}

export interface DailyEventObjectAvailableDevicesUpdated
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'available-devices-updated'>;
  availableDevices: MediaDeviceInfo[];
}

export interface DailyEventObjectShowLocalVideoChanged
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'show-local-video-changed'>;
  show: boolean;
}
export interface DailyEventObjectInputSettingsUpdated
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'input-settings-updated'>;
  inputSettings: DailyInputSettings;
}

export interface DailyEventObjectSendSettingsUpdated
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'send-settings-updated'>;
  sendSettings: DailySendSettings;
}

export interface DailyEventObjectLocalAudioLevel extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'local-audio-level'>;
  audioLevel: number;
}

export interface DailyEventObjectRemoteParticipantsAudioLevel
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'remote-participants-audio-level'>;
  participantsAudioLevel: DailyParticipantsAudioLevel;
}

export interface DailyEventObjectLiveStreamingStarted
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'live-streaming-started'>;
  layout?: DailyLiveStreamingLayoutConfig<'start'>;
  instanceId?: string;
}
export interface DailyEventObjectLiveStreamingUpdated
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'live-streaming-updated'>;
  endpoint?: DailyStreamingEndpoint;
  state: DailyStreamingState;
  instanceId?: string;
}

export interface DailyEventObjectLiveStreamingStopped
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'live-streaming-stopped'>;
  instanceId?: string;
}

export interface DailyEventObjectTranscriptionStarted
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'transcription-started'>;
  instanceId: string;
  transcriptId?: string;
  language: string;
  model: string;
  tier?: string;
  profanity_filter?: boolean;
  redact?: Array<string> | Array<boolean> | boolean;
  endpointing?: number | boolean;
  punctuate?: boolean;
  extra?: Record<string, any>;
  includeRawResponse?: boolean;
  startedBy: string;
}

export interface DailyEventObjectTranscriptionStopped
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'transcription-stopped'>;
  instanceId: string;
  updatedBy: string;
}

export interface DailyEventObjectTranscriptionError
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'transcription-error'>;
  instanceId: string;
  errorMsg?: string;
}

export type DailyRemoteMediaPlayerStopReason =
  | DailyRemoteMediaPlayerEOS
  | DailyRemoteMediaPlayerPeerStopped;

export interface DailyEventObjectRemoteMediaPlayerUpdate
  extends DailyEventObjectBase {
  action: Extract<
    DailyEvent,
    'remote-media-player-started' | 'remote-media-player-updated'
  >;
  updatedBy: string;
  session_id: string;
  remoteMediaPlayerState: DailyRemoteMediaPlayerState;
}

export interface DailyEventObjectRemoteMediaPlayerStopped
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'remote-media-player-stopped'>;
  session_id: string;
  updatedBy: string;
  reason: DailyRemoteMediaPlayerStopReason;
}

export interface DailyEventObjectCustomButtonClick
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'custom-button-click'>;
  button_id: string;
}

export interface DailyEventObjectSelectedDevicesUpdated
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'selected-devices-updated'>;
  devices: DailyDeviceInfos;
}

export interface DailyEventObjectSidebarViewChanged
  extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'sidebar-view-changed'>;
  view: SidebarView;
}

export interface DailyEventObjectDialinConnected extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'dialin-connected'>;
  sipHeaders?: Record<string, any>;
  sipFrom?: string;
  actionTraceId?: string;
}

export interface DailyEventObjectDialinError extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'dialin-error'>;
  errorMsg: string;
  actionTraceId?: string;
}

export interface DailyEventObjectDialinStopped extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'dialin-stopped'>;
  sipHeaders?: Record<string, any>;
  sipFrom?: string;
  actionTraceId?: string;
}

export interface DailyEventObjectDialinWarning extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'dialin-warning'>;
  errorMsg: string;
  actionTraceId?: string;
}

export interface DailyEventObjectDialOutConnected extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'dialout-connected'>;
  sessionId?: string;
  userId?: string;
  actionTraceId?: string;
}

export interface DailyEventObjectDialOutError extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'dialout-error'>;
  errorMsg: string;
  sessionId?: string;
  userId?: string;
  actionTraceId?: string;
}

export interface DailyEventObjectDialOutStopped extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'dialout-stopped'>;
  sessionId?: string;
  userId?: string;
  actionTraceId?: string;
}

export interface DailyEventObjectDialOutWarning extends DailyEventObjectBase {
  action: Extract<DailyEvent, 'dialout-warning'>;
  errorMsg: string;
  sessionId?: string;
  actionTraceId?: string;
}

export type DailyEventObject<T extends DailyEvent = any> =
  T extends DailyEventObjectAppMessage['action']
    ? DailyEventObjectAppMessage
    : T extends DailyEventObjectNoPayload['action']
    ? DailyEventObjectNoPayload
    : T extends DailyEventObjectCameraError['action']
    ? DailyEventObjectCameraError
    : T extends DailyEventObjectFatalError['action']
    ? DailyEventObjectFatalError
    : T extends DailyEventObjectNonFatalError['action']
    ? DailyEventObjectNonFatalError
    : T extends DailyEventObjectGenericError['action']
    ? DailyEventObjectGenericError
    : T extends DailyEventObjectParticipants['action']
    ? DailyEventObjectParticipants
    : T extends DailyEventObjectLiveStreamingStarted['action']
    ? DailyEventObjectLiveStreamingStarted
    : T extends DailyEventObjectLiveStreamingUpdated['action']
    ? DailyEventObjectLiveStreamingUpdated
    : T extends DailyEventObjectLiveStreamingStopped['action']
    ? DailyEventObjectLiveStreamingStopped
    : T extends DailyEventObjectLiveStreamingError['action']
    ? DailyEventObjectLiveStreamingError
    : T extends DailyEventObjectTranscriptionStarted['action']
    ? DailyEventObjectTranscriptionStarted
    : T extends DailyEventObjectTranscriptionMessage['action']
    ? DailyEventObjectTranscriptionMessage
    : T extends DailyEventObjectTranscriptionStopped['action']
    ? DailyEventObjectTranscriptionStopped
    : T extends DailyEventObjectTranscriptionError['action']
    ? DailyEventObjectTranscriptionError
    : T extends DailyEventObjectParticipant['action']
    ? DailyEventObjectParticipant
    : T extends DailyEventObjectParticipantLeft['action']
    ? DailyEventObjectParticipantLeft
    : T extends DailyEventObjectParticipantCounts['action']
    ? DailyEventObjectParticipantCounts
    : T extends DailyEventObjectWaitingParticipant['action']
    ? DailyEventObjectWaitingParticipant
    : T extends DailyEventObjectAccessState['action']
    ? DailyEventObjectAccessState
    : T extends DailyEventObjectMeetingSessionUpdated['action']
    ? DailyEventObjectMeetingSessionUpdated
    : T extends DailyEventObjectMeetingSessionStateUpdated['action']
    ? DailyEventObjectMeetingSessionStateUpdated
    : T extends DailyEventObjectTrack['action']
    ? DailyEventObjectTrack
    : T extends DailyEventObjectRecordingStarted['action']
    ? DailyEventObjectRecordingStarted
    : T extends DailyEventObjectRecordingStopped['action']
    ? DailyEventObjectRecordingStopped
    : T extends DailyEventObjectRecordingError['action']
    ? DailyEventObjectRecordingError
    : T extends DailyEventObjectRecordingData['action']
    ? DailyEventObjectRecordingData
    : T extends DailyEventObjectRemoteMediaPlayerUpdate['action']
    ? DailyEventObjectRemoteMediaPlayerUpdate
    : T extends DailyEventObjectRemoteMediaPlayerStopped['action']
    ? DailyEventObjectRemoteMediaPlayerStopped
    : T extends DailyEventObjectMouseEvent['action']
    ? DailyEventObjectMouseEvent
    : T extends DailyEventObjectTouchEvent['action']
    ? DailyEventObjectTouchEvent
    : T extends DailyEventObjectNetworkQualityEvent['action']
    ? DailyEventObjectNetworkQualityEvent
    : T extends DailyEventObjectCpuLoadEvent['action']
    ? DailyEventObjectCpuLoadEvent
    : T extends DailyEventObjectFaceCounts['action']
    ? DailyEventObjectFaceCounts
    : T extends DailyEventObjectNetworkConnectionEvent['action']
    ? DailyEventObjectNetworkConnectionEvent
    : T extends DailyEventObjectTestCompleted['action']
    ? DailyEventObjectTestCompleted
    : T extends DailyEventObjectActiveSpeakerChange['action']
    ? DailyEventObjectActiveSpeakerChange
    : T extends DailyEventObjectActiveSpeakerModeChange['action']
    ? DailyEventObjectActiveSpeakerModeChange
    : T extends DailyEventObjectLangUpdated['action']
    ? DailyEventObjectLangUpdated
    : T extends DailyEventObjectThemeUpdated['action']
    ? DailyEventObjectThemeUpdated
    : T extends DailyEventObjectReceiveSettingsUpdated['action']
    ? DailyEventObjectReceiveSettingsUpdated
    : T extends DailyEventObjectAvailableDevicesUpdated['action']
    ? DailyEventObjectAvailableDevicesUpdated
    : T extends DailyEventObjectShowLocalVideoChanged['action']
    ? DailyEventObjectShowLocalVideoChanged
    : T extends DailyEventObjectInputSettingsUpdated['action']
    ? DailyEventObjectInputSettingsUpdated
    : T extends DailyEventObjectSendSettingsUpdated['action']
    ? DailyEventObjectSendSettingsUpdated
    : T extends DailyEventObjectCustomButtonClick['action']
    ? DailyEventObjectCustomButtonClick
    : T extends DailyEventObjectSelectedDevicesUpdated['action']
    ? DailyEventObjectSelectedDevicesUpdated
    : T extends DailyEventObjectSidebarViewChanged['action']
    ? DailyEventObjectSidebarViewChanged
    : T extends DailyEventObjectDialinConnected['action']
    ? DailyEventObjectDialinConnected
    : T extends DailyEventObjectDialinError['action']
    ? DailyEventObjectDialinError
    : T extends DailyEventObjectDialinStopped['action']
    ? DailyEventObjectDialinStopped
    : T extends DailyEventObjectDialinWarning['action']
    ? DailyEventObjectDialinWarning
    : T extends DailyEventObjectDialOutConnected['action']
    ? DailyEventObjectDialOutConnected
    : T extends DailyEventObjectDialOutError['action']
    ? DailyEventObjectDialOutError
    : T extends DailyEventObjectDialOutStopped['action']
    ? DailyEventObjectDialOutStopped
    : T extends DailyEventObjectDialOutWarning['action']
    ? DailyEventObjectDialOutWarning
    : T extends DailyEventObjectLocalAudioLevel['action']
    ? DailyEventObjectLocalAudioLevel
    : T extends DailyEventObjectRemoteParticipantsAudioLevel['action']
    ? DailyEventObjectRemoteParticipantsAudioLevel
    : T extends DailyEvent
    ? DailyEventObjectBase
    : any;

export interface DailyCallFactory {
  createCallObject(properties?: DailyFactoryOptions): DailyCall;
  wrap(iframe: HTMLIFrameElement, properties?: DailyFactoryOptions): DailyCall;
  createFrame(
    parentElement: HTMLElement,
    properties?: DailyFactoryOptions
  ): DailyCall;
  createFrame(properties?: DailyFactoryOptions): DailyCall;
  createTransparentFrame(properties?: DailyFactoryOptions): DailyCall;
  getCallInstance(callClientId?: string): DailyCall | undefined;
}

export interface DailyCallStaticUtils {
  supportedBrowser(): DailyBrowserInfo;
  version(): string;
}

export type DailyAccess = 'unknown' | SpecifiedDailyAccess;

export type SpecifiedDailyAccess = { level: 'none' | 'lobby' | 'full' };

export type DailyAccessState = {
  access: DailyAccess;
  awaitingAccess?: SpecifiedDailyAccess;
};

export type DailyAccessRequest = {
  access?: { level: 'full' };
  name: string;
};

type DailyStreamingParticipantsSortMethod = 'active';

export interface DailyStreamingParticipantsConfig {
  video?: string[];
  audio?: string[];
  sort?: DailyStreamingParticipantsSortMethod;
}

export interface DailyStreamingDefaultLayoutConfig {
  preset: 'default';
  max_cam_streams?: number;
  participants?: DailyStreamingParticipantsConfig;
}

export interface DailyStreamingSingleParticipantLayoutConfig {
  preset: 'single-participant';
  session_id: string;
}

export interface DailyStreamingActiveParticipantLayoutConfig {
  preset: 'active-participant';
  participants?: DailyStreamingParticipantsConfig;
}

export interface DailyStreamingAudioOnlyLayoutConfig {
  preset: 'audio-only';
  participants?: DailyStreamingParticipantsConfig;
}

export type DailyStreamingPortraitLayoutVariant = 'vertical' | 'inset';

export interface DailyStreamingPortraitLayoutConfig {
  preset: 'portrait';
  variant?: DailyStreamingPortraitLayoutVariant;
  max_cam_streams?: number;
  participants?: DailyStreamingParticipantsConfig;
}

export interface DailyUpdateStreamingCustomLayoutConfig {
  preset: 'custom';
  participants?: DailyStreamingParticipantsConfig;
  composition_params?: {
    [key: string]: boolean | number | string;
  };
}

export interface DailyStartStreamingCustomLayoutConfig
  extends DailyUpdateStreamingCustomLayoutConfig {
  composition_id?: string;
  session_assets?: {
    [key: string]: string;
  };
}

type DailyStreamingLayoutConfigType = 'start' | 'update';
type DailyStartStreamingMethod = 'liveStreaming' | 'recording';

export type DailyStreamingLayoutConfig<
  Type extends DailyStreamingLayoutConfigType = 'start'
> =
  | DailyStreamingDefaultLayoutConfig
  | DailyStreamingSingleParticipantLayoutConfig
  | DailyStreamingActiveParticipantLayoutConfig
  | DailyStreamingPortraitLayoutConfig
  | DailyStreamingAudioOnlyLayoutConfig
  | (Type extends 'start'
      ? DailyStartStreamingCustomLayoutConfig
      : DailyUpdateStreamingCustomLayoutConfig);

export type DailyLiveStreamingLayoutConfig<
  Type extends DailyStreamingLayoutConfigType = 'start'
> = Exclude<
  DailyStreamingLayoutConfig<Type>,
  DailyStreamingAudioOnlyLayoutConfig
>;

export type DailyStreamingState = 'connected' | 'interrupted';

export type DailyRemoteMediaPlayerSettingPlay = 'play';
export type DailyRemoteMediaPlayerSettingPause = 'pause';

export type DailyRemoteMediaPlayerStatePlaying = 'playing';
export type DailyRemoteMediaPlayerStatePaused = 'paused';
export type DailyRemoteMediaPlayerStateBuffering = 'buffering';

export type DailyRemoteMediaPlayerEOS = 'EOS';
export type DailyRemoteMediaPlayerPeerStopped = 'stopped-by-peer';

export interface DailyStreamingOptions<
  Method extends DailyStartStreamingMethod,
  Type extends DailyStreamingLayoutConfigType = 'start'
> {
  width?: number;
  height?: number;
  fps?: number;
  videoBitrate?: number;
  audioBitrate?: number;
  minIdleTimeOut?: number;
  maxDuration?: number;
  backgroundColor?: string;
  instanceId?: string;
  layout?: Method extends 'recording'
    ? DailyStreamingLayoutConfig<Type>
    : DailyLiveStreamingLayoutConfig<Type>;
}

export interface DailyStreamingEndpoint {
  endpoint: string;
}

export interface DailyLiveStreamingOptions<
  Type extends DailyStreamingLayoutConfigType = 'start'
> extends DailyStreamingOptions<'liveStreaming', Type> {
  rtmpUrl?: string | string[];
  endpoints?: DailyStreamingEndpoint[];
}

export interface RemoteMediaPlayerSimulcastEncoding {
  maxBitrate: number;
  maxFramerate?: number;
  scaleResolutionDownBy?: number;
}

export interface DailyRemoteMediaPlayerSettings {
  state?:
    | DailyRemoteMediaPlayerSettingPlay
    | DailyRemoteMediaPlayerSettingPause;
  volume?: number;
  simulcastEncodings?: RemoteMediaPlayerSimulcastEncoding[];
}

export interface DailyRemoteMediaPlayerStartOptions {
  url: string;
  settings?: DailyRemoteMediaPlayerSettings;
}

export interface DailyRemoteMediaPlayerUpdateOptions {
  session_id: string;
  settings: DailyRemoteMediaPlayerSettings;
}

export interface DailyRemoteMediaPlayerState {
  state:
    | DailyRemoteMediaPlayerStatePlaying
    | DailyRemoteMediaPlayerStatePaused
    | DailyRemoteMediaPlayerStateBuffering;
  settings: DailyRemoteMediaPlayerSettings;
}

export interface DailyRemoteMediaPlayerInfo {
  session_id: string;
  remoteMediaPlayerState: DailyRemoteMediaPlayerState;
}

export interface DailyTranscriptionDeepgramOptions {
  language?: string;
  model?: string;
  tier?: string;
  profanity_filter?: boolean;
  redact?: Array<string> | Array<boolean> | boolean;
  endpointing?: number | boolean;
  punctuate?: boolean;
  extra?: Record<string, any>;
  includeRawResponse?: boolean;
  instanceId?: string;
  participants?: Array<string>;
}

export interface DailyTranscriptionUpdateOptions {
  instanceId?: string;
  participants: Array<string>;
}

export interface DailyTranscriptionStopOptions {
  instanceId?: string;
}
export type SidebarView =
  | null
  | 'people'
  | 'chat'
  | 'network'
  | 'breakout'
  | string;

export type DailyDialOutAudioCodecs = 'PCMU' | 'OPUS' | 'G722' | 'PCMA';

export type DailyDialOutVideoCodecs = 'H264' | 'VP8';

export interface DailyDialOutCodecs {
  audio?: Array<DailyDialOutAudioCodecs>;
  video?: Array<DailyDialOutVideoCodecs>;
}

export interface DailyDialOutSession {
  sessionId: string;
}

export interface DailyStartDialoutSipOptions {
  sipUri?: string;
  displayName?: string;
  userId?: string;
  video?: boolean;
  codecs?: DailyDialOutCodecs;
}

export interface DailyStartDialoutPhoneOptions {
  phoneNumber?: string;
  displayName?: string;
  userId?: string;
  codecs?: DailyDialOutCodecs;
  callerId?: string;
}

export type DailyStartDialoutOptions =
  | DailyStartDialoutSipOptions
  | DailyStartDialoutPhoneOptions;

export interface DailyScreenShareUpdateOptions {
  screenVideo: {
    enabled: boolean;
  };
  screenAudio: {
    enabled: boolean;
  };
}

export type DailyCameraFacingMode = 'user' | 'environment' | undefined;

export interface DailyMediaDeviceInfo extends MediaDeviceInfo {
  facing?: DailyCameraFacingMode;
}

export interface DailySipCallTransferOptions {
  sessionId: string;
  toEndPoint: string;
}

export interface DailySipReferOptions {
  sessionId: string;
  toEndPoint: string;
}

export interface DailyCall {
  callClientId: string;
  iframe(): HTMLIFrameElement | null;
  join(properties?: DailyCallOptions): Promise<DailyParticipantsObject | void>;
  leave(): Promise<void>;
  destroy(): Promise<void>;
  isDestroyed(): boolean;
  loadCss(properties: {
    bodyClass?: string;
    cssFile?: string;
    cssText?: string;
  }): DailyCall;
  meetingState(): DailyMeetingState;
  accessState(): DailyAccessState;
  participants(): DailyParticipantsObject;
  participantCounts(): DailyParticipantCounts;
  updateParticipant(
    sessionId: string,
    updates: DailyParticipantUpdateOptions
  ): DailyCall;
  updateParticipants(updates: {
    [sessionId: string]: DailyParticipantUpdateOptions;
  }): DailyCall;
  waitingParticipants(): { [id: string]: DailyWaitingParticipant };
  updateWaitingParticipant(
    id: string,
    updates: DailyWaitingParticipantUpdateOptions
  ): Promise<{ id: string }>;
  updateWaitingParticipants(updates: {
    [id: string]: DailyWaitingParticipantUpdateOptions;
  }): Promise<{ ids: string[] }>;
  requestAccess(
    access: DailyAccessRequest
  ): Promise<{ access: DailyAccess; granted: boolean }>;
  localAudio(): boolean;
  localVideo(): boolean;
  setLocalAudio(
    enabled: boolean,
    options?: { forceDiscardTrack: boolean }
  ): DailyCall;
  setLocalVideo(enabled: boolean): DailyCall;
  localScreenAudio(): boolean;
  localScreenVideo(): boolean;
  updateScreenShare(options?: DailyScreenShareUpdateOptions): void;
  getReceiveSettings(
    id: string,
    options?: { showInheritedValues: boolean }
  ): Promise<DailySingleParticipantReceiveSettings>;
  getReceiveSettings(): Promise<DailyReceiveSettings>;
  updateReceiveSettings(
    receiveSettings: DailyReceiveSettingsUpdates
  ): Promise<DailyReceiveSettings>;
  updateInputSettings(
    inputSettings: DailyInputSettings
  ): Promise<{ inputSettings: DailyInputSettings }>;
  getInputSettings(): Promise<DailyInputSettings>;
  updateCustomTrayButtons(customTrayButtons: DailyCustomTrayButtons): DailyCall;
  customTrayButtons(): DailyCustomTrayButtons;
  setCustomIntegrations(customIntegrations: DailyCustomIntegrations): DailyCall;
  customIntegrations(): DailyCustomIntegrations;
  startCustomIntegrations(id: string | string[]): DailyCall;
  stopCustomIntegrations(id: string | string[]): DailyCall;
  setBandwidth(bw: {
    kbs?: number | 'NO_CAP' | null;
    trackConstraints?: MediaTrackConstraints;
  }): DailyCall;
  getDailyLang(): Promise<{
    lang: DailyLanguage;
    langSetting: DailyLanguageSetting;
  }>;
  setDailyLang(lang: DailyLanguageSetting): DailyCall;
  setProxyUrl(proxyUrl?: string): DailyCall;
  setIceConfig(iceConfig?: DailyIceConfig): DailyCall;
  /**
   * @deprecated This function will be removed. Use the method meetingSessionSummary() instead.
   */
  getMeetingSession(): Promise<{
    meetingSession: DailyMeetingSession;
  }>;
  meetingSessionSummary(): DailyMeetingSessionSummary;
  meetingSessionState(): DailyMeetingSessionState;
  setMeetingSessionData(
    data: unknown,
    mergeStrategy?: DailySessionDataMergeStrategy
  ): void;
  setUserName(
    name: string,
    options?: { thisMeetingOnly?: boolean }
  ): Promise<{ userName: string }>;
  setUserData(data: unknown): Promise<{ userData: unknown }>;
  startCamera(properties?: DailyCallOptions): Promise<DailyDeviceInfos>;
  startLocalAudioLevelObserver(interval?: number): Promise<void>;
  isLocalAudioLevelObserverRunning(): boolean;
  stopLocalAudioLevelObserver(): void;
  getLocalAudioLevel(): number;
  startRemoteParticipantsAudioLevelObserver(interval?: number): Promise<void>;
  isRemoteParticipantsAudioLevelObserverRunning(): boolean;
  stopRemoteParticipantsAudioLevelObserver(): void;
  getRemoteParticipantsAudioLevel(): DailyParticipantsAudioLevel;
  cycleCamera(properties?: {
    preferDifferentFacingMode?: boolean;
  }): Promise<{ device?: MediaDeviceInfo | null }>;
  cycleMic(): Promise<{ device?: MediaDeviceInfo | null }>;
  startCustomTrack(properties: StartCustomTrackOptions): Promise<string>;
  stopCustomTrack(trackName: string): Promise<string>;
  setInputDevicesAsync(devices: {
    audioDeviceId?: string | false | null;
    audioSource?: MediaStreamTrack | false;
    videoDeviceId?: string | false | null;
    videoSource?: MediaStreamTrack | false;
  }): Promise<DailyDeviceInfos>;
  setOutputDeviceAsync(audioDevice: {
    outputDeviceId?: string;
  }): Promise<DailyDeviceInfos>;
  getInputDevices(): Promise<DailyDeviceInfos>;
  preAuth(properties?: DailyCallOptions): Promise<{ access: DailyAccess }>;
  load(properties?: DailyLoadOptions): Promise<void>;
  startScreenShare(properties?: DailyStartScreenShareOptions): void;
  stopScreenShare(): void;
  startRecording(options?: DailyStreamingOptions<'recording', 'start'>): void;
  updateRecording(options: {
    layout?: DailyStreamingLayoutConfig<'update'>;
    instanceId?: string;
  }): void;
  stopRecording(options?: { instanceId: string }): void;
  startLiveStreaming(options: DailyLiveStreamingOptions<'start'>): void;
  updateLiveStreaming(options: {
    layout?: DailyLiveStreamingLayoutConfig<'update'>;
    instanceId?: string;
  }): void;
  addLiveStreamingEndpoints(options: {
    endpoints: DailyStreamingEndpoint[];
    instanceId?: string;
  }): void;
  removeLiveStreamingEndpoints(options: {
    endpoints: DailyStreamingEndpoint[];
    instanceId?: string;
  }): void;
  stopLiveStreaming(options?: { instanceId: string }): void;
  startRemoteMediaPlayer(
    options: DailyRemoteMediaPlayerStartOptions
  ): Promise<DailyRemoteMediaPlayerInfo>;
  stopRemoteMediaPlayer(session_id: string): Promise<void>;
  updateRemoteMediaPlayer(
    options: DailyRemoteMediaPlayerUpdateOptions
  ): Promise<DailyRemoteMediaPlayerInfo>;
  startTranscription(options?: DailyTranscriptionDeepgramOptions): void;
  updateTranscription(options: DailyTranscriptionUpdateOptions): void;
  stopTranscription(options?: DailyTranscriptionStopOptions): void;
  testCallQuality(): Promise<DailyCallQualityTestResults>;
  stopTestCallQuality(): void;
  testPeerToPeerCallQuality(options: {
    videoTrack: MediaStreamTrack;
    duration?: number;
  }): Promise<DailyP2PCallQualityTestResults>;
  stopTestPeerToPeerCallQuality(): void;
  testWebsocketConnectivity(): Promise<DailyWebsocketConnectivityTestResults>;
  abortTestWebsocketConnectivity(): void;
  testNetworkConnectivity(
    videoTrack: MediaStreamTrack
  ): Promise<DailyNetworkConnectivityTestStats>;
  abortTestNetworkConnectivity(): void;
  /**
   * @deprecated This function will be removed. Use the method
   *    testPeerToPeerCallQuality() instead.
   */
  testConnectionQuality(options: {
    videoTrack: MediaStreamTrack;
    duration?: number;
  }): Promise<DailyConnectionQualityTestStats>;
  /**
   * @deprecated This function will be removed. Use the method
   *    stopTestPeerToPeerCallQuality() instead.
   */
  stopTestConnectionQuality(): void;
  getNetworkStats(): Promise<DailyNetworkStats>;
  getCpuLoadStats(): Promise<DailyCpuLoadStats>;
  updateSendSettings(settings: DailySendSettings): Promise<DailySendSettings>;
  getSendSettings(): DailySendSettings | null;
  getActiveSpeaker(): { peerId?: string };
  setActiveSpeakerMode(enabled: boolean): DailyCall;
  activeSpeakerMode(): boolean;
  subscribeToTracksAutomatically(): boolean;
  setSubscribeToTracksAutomatically(enabled: boolean): DailyCall;
  enumerateDevices(): Promise<{ devices: DailyMediaDeviceInfo[] }>;
  sendAppMessage(data: any, to?: string | string[]): DailyCall;
  addFakeParticipant(details?: { aspectRatio: number }): DailyCall;
  setShowNamesMode(mode: false | 'always' | 'never'): DailyCall;
  setShowLocalVideo(show: boolean): DailyCall;
  setShowParticipantsBar(show: boolean): DailyCall;
  theme(): DailyThemeConfig;
  setTheme(theme: DailyThemeConfig): Promise<DailyThemeConfig>;
  showLocalVideo(): boolean;
  showParticipantsBar(): boolean;
  requestFullscreen(): Promise<void>;
  exitFullscreen(): void;
  room(options?: {
    includeRoomConfigDefaults: boolean;
  }): Promise<DailyPendingRoomInfo | DailyRoomInfo | null>;
  geo(): Promise<{ current: string }>;
  getNetworkTopology(): Promise<{
    topology: DailyNetworkTopology | 'none';
    error?: string;
  }>;
  setNetworkTopology(options: {
    topology: DailyNetworkTopology;
  }): Promise<{ workerId?: string; error?: string }>;
  setPlayNewParticipantSound(sound: boolean | number): void;
  getSidebarView(): Promise<SidebarView>;
  setSidebarView(view: SidebarView): DailyCall;
  on<T extends DailyEvent>(
    event: T,
    handler: (event: DailyEventObject<T>) => void
  ): DailyCall;
  once<T extends DailyEvent>(
    event: T,
    handler: (event: DailyEventObject<T>) => void
  ): DailyCall;
  off<T extends DailyEvent>(
    event: T,
    handler: (event: DailyEventObject<T>) => void
  ): DailyCall;
  properties: {
    dailyConfig?: DailyAdvancedConfig;
    userName?: string;
  };
  startDialOut(
    options: DailyStartDialoutOptions
  ): Promise<{ session?: DailyDialOutSession }>;
  stopDialOut(options: { sessionId: string }): Promise<void>;
  sendDTMF(options: { sessionId: string; tones: string }): Promise<void>;
  sipCallTransfer(options: DailySipCallTransferOptions): Promise<void>;
  sipRefer(options: DailySipReferOptions): Promise<void>;
}

declare const Daily: DailyCallFactory & DailyCallStaticUtils;

export default Daily;
