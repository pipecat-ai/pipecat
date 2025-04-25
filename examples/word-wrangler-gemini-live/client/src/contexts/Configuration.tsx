import React, { createContext, useContext, useState, ReactNode } from 'react';
import { PersonalityType, DEFAULT_PERSONALITY } from '@/types/personality';

interface ConfigurationContextProps {
  personality: PersonalityType;
  setPersonality: (personality: PersonalityType) => void;
}

const ConfigurationContext = createContext<
  ConfigurationContextProps | undefined
>(undefined);

interface ConfigurationProviderProps {
  children: ReactNode;
}

export function ConfigurationProvider({
  children,
}: ConfigurationProviderProps) {
  const [personality, setPersonality] =
    useState<PersonalityType>(DEFAULT_PERSONALITY);

  const value = {
    personality,
    setPersonality,
  };

  return (
    <ConfigurationContext.Provider value={value}>
      {children}
    </ConfigurationContext.Provider>
  );
}

export function useConfigurationSettings() {
  const context = useContext(ConfigurationContext);
  if (context === undefined) {
    throw new Error(
      'useConfigurationSettings must be used within a ConfigurationProvider'
    );
  }
  return context;
}
