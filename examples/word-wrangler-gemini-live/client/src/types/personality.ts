export type PersonalityType =
  | 'friendly'
  | 'professional'
  | 'enthusiastic'
  | 'thoughtful'
  | 'witty';

// This object can be useful for displaying user-friendly labels or descriptions
export const PERSONALITY_PRESETS: Record<PersonalityType, string> = {
  friendly: 'Friendly',
  professional: 'Professional',
  enthusiastic: 'Enthusiastic',
  thoughtful: 'Thoughtful',
  witty: 'Witty',
};

// Default personality to use
export const DEFAULT_PERSONALITY: PersonalityType = 'witty';
