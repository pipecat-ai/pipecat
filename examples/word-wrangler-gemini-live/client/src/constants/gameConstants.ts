// Game configuration
export const GAME_CONFIG = {
  MAX_SKIPS: 3,
  GAME_DURATION: 60, // seconds
  WORD_POOL_SIZE: 30,
  ANIMATION_DURATION: 1000, // ms
  TIMER_INTERVAL: 1000, // ms
  LOW_TIME_WARNING: 10, // seconds
};

// Game states
export const GAME_STATES = {
  IDLE: "idle",
  CONNECTING: "connecting",
  WAITING_FOR_INTRO: "waitingForIntro",
  ACTIVE: "active",
  FINISHED: "finished",
} as const;

export type GameState = (typeof GAME_STATES)[keyof typeof GAME_STATES];

// Text used in the game
export const GAME_TEXT = {
  time: "Time",
  score: "Score",
  gameOver: "Game Over!",
  finalScore: "Final Score",
  correct: "Mark Correct",
  skip: "Skip →",
  noSkips: "No Skips Left",
  skipsRemaining: (num: number) => `Skip (${num} left)`,
  startingGame: `How many words can you describe in ${GAME_CONFIG.GAME_DURATION} seconds?`,
  waitingForIntro: "Getting ready...",
  clickToStart: "Press Start Game to begin",
  describeWord: "Describe the following word:",
  introTitle: "How many words can you describe within 60 seconds?",
  introGuide1: "Earn points each time the AI correctly guesses the word",
  introGuide2: "Do not say the word, or you will lose points",
  introGuide3: "You can skip the word if you don't know it",
  aiPersonality: "AI Personality",
  finalScoreMessage: "Your best score:",
};

// Pattern for detecting guesses in transcripts
export const TRANSCRIPT_PATTERNS = {
  // Match both "Is it "word"?" and "Is it a/an word?" patterns
  GUESS_PATTERN:
    /is it [""]?([^""?]+)[""]?(?:\?)?|is it (?:a|an) ([^?]+)(?:\?)?/i,
};

// Connection states
export const CONNECTION_STATES = {
  ACTIVE: ["connected", "ready"],
  CONNECTING: ["connecting", "initializing", "initialized", "authenticating"],
  DISCONNECTING: ["disconnecting"],
};

// Button text
export const BUTTON_TEXT = {
  START: "Start Game",
  END: "End Game",
  CONNECTING: "Connecting...",
  STARTING: "Starting...",
  RESTART: "Play Again",
};
